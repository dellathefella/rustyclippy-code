//! Web tools: `websearch` (DuckDuckGo / Brave / Tavily / SearXNG) and `webfetch`.

use std::net::{IpAddr, Ipv4Addr};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use regex::Regex;
use rustyclippy_core::config::WebProvider;
use serde_json::{json, Value};

use crate::registry::{Tool, ToolContext, ToolResult};

#[derive(Debug, Clone)]
struct SearchResult {
    title: String,
    url: String,
    snippet: String,
}

pub struct WebSearchTool;

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &'static str {
        "websearch"
    }

    fn description(&self) -> &'static str {
        "Search the web and return titles, URLs and snippets. \
         Use for current documentation, release notes, or errors you cannot resolve locally."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "description": "Maximum results (default from config)."}
            },
            "required": ["query"]
        })
    }

    fn mutates(&self) -> bool {
        false
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let query = args
            .get("query")
            .and_then(|value| value.as_str())
            .context("websearch requires a `query` string")?;
        let max_results = args
            .get("max_results")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .unwrap_or(ctx.web.max_results)
            .clamp(1, 20);

        let results = match ctx.web.provider {
            WebProvider::DuckDuckGo => duckduckgo(query, max_results).await?,
            WebProvider::Brave => brave(query, max_results, ctx.web.api_key()).await?,
            WebProvider::Tavily => tavily(query, max_results, ctx.web.api_key()).await?,
            WebProvider::Searxng => {
                searxng(query, max_results, ctx.web.searxng_url.as_deref()).await?
            }
        };

        if results.is_empty() {
            return Ok(ToolResult::success(format!(
                "no results for `{query}` (provider: {:?})",
                ctx.web.provider
            )));
        }

        let mut out = String::new();
        for (index, result) in results.iter().enumerate() {
            out.push_str(&format!(
                "{}. {}\n   {}\n   {}\n",
                index + 1,
                result.title,
                result.url,
                result.snippet
            ));
        }
        Ok(ToolResult::success(ctx.truncate(&out)))
    }
}

pub struct WebFetchTool;

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &'static str {
        "webfetch"
    }

    fn description(&self) -> &'static str {
        "Fetch a URL and return its readable text (HTML tags stripped)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP(S) URL."},
                "max_chars": {"type": "integer", "description": "Maximum characters to return."}
            },
            "required": ["url"]
        })
    }

    fn mutates(&self) -> bool {
        false
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let url = args
            .get("url")
            .and_then(|value| value.as_str())
            .context("webfetch requires a `url` string")?;
        if !url.starts_with("http://") && !url.starts_with("https://") {
            bail!("url must start with http:// or https://");
        }
        // Refuse loopback, link-local and private addresses. Without this the
        // model could read cloud metadata (169.254.169.254) or probe the local
        // inference server through the fetch tool.
        guard_url(url)?;
        let max_chars = args
            .get("max_chars")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .unwrap_or(ctx.web.fetch_max_chars);

        let client = http_client()?;
        let response = client
            .get(url)
            .send()
            .await
            .with_context(|| format!("GET {url}"))?;
        let status = response.status();
        if !status.is_success() {
            return Ok(ToolResult::failure(format!("HTTP {status} for {url}")));
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        let body = response.text().await.context("reading response body")?;

        let text = if content_type.contains("html") || body.trim_start().starts_with('<') {
            strip_html(&body)
        } else {
            body
        };

        let trimmed = collapse_whitespace(&text);
        Ok(ToolResult::success(truncate_chars(&trimmed, max_chars)))
    }
}

/// Reject URLs that point at loopback, link-local, or private addresses.
///
/// The fetch tool is reachable by the model, so without this guard a prompt
/// could read cloud metadata (`169.254.169.254`), probe the local inference
/// server, or scan the LAN. Hostnames are resolved first so a DNS name that
/// maps to a private address is caught too.
fn guard_url(url: &str) -> Result<()> {
    let parsed = reqwest::Url::parse(url).with_context(|| format!("parsing url `{url}`"))?;
    let host = parsed
        .host_str()
        .with_context(|| format!("url `{url}` has no host"))?;

    // A literal IP is checked directly; a name is resolved and every address
    // must pass, so a DNS rebind cannot slip through.
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_blocked_ip(ip) {
            bail!("refusing to fetch `{url}`: {ip} is a private or loopback address");
        }
        return Ok(());
    }

    let port = parsed.port_or_known_default().unwrap_or(80);
    let addresses = std::net::ToSocketAddrs::to_socket_addrs(&(host, port))
        .with_context(|| format!("resolving host `{host}`"))?;
    for address in addresses {
        if is_blocked_ip(address.ip()) {
            bail!(
                "refusing to fetch `{url}`: `{host}` resolves to {} (private or loopback)",
                address.ip()
            );
        }
    }
    Ok(())
}

/// True for addresses a fetch tool must never reach.
fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_blocked_v4(v4),
        IpAddr::V6(v6) => {
            // IPv4-mapped addresses (`::ffff:127.0.0.1`) must be checked as v4.
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_blocked_v4(v4);
            }
            v6.is_loopback()
                || v6.is_unspecified()
                || (v6.segments()[0] & 0xffc0) == 0xfe80 // link-local fe80::/10
                || (v6.segments()[0] & 0xfe00) == 0xfc00 // unique-local fc00::/7
        }
    }
}

fn is_blocked_v4(ip: Ipv4Addr) -> bool {
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local() // 169.254.0.0/16, cloud metadata
        || ip.is_broadcast()
        || ip.is_unspecified()
        // 0.0.0.0/8 "this network"
        || ip.octets()[0] == 0
        // 100.64.0.0/10 carrier-grade NAT
        || (ip.octets()[0] == 100 && (64..128).contains(&ip.octets()[1]))
}

fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60))
        .user_agent("Mozilla/5.0 (compatible; rustyclippy-code/0.1)")
        .build()
        .context("building HTTP client")
}

async fn duckduckgo(query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
    let client = http_client()?;
    let url = format!(
        "https://html.duckduckgo.com/html/?q={}",
        percent_encode(query)
    );
    let response = client.get(url).send().await.context("DuckDuckGo request")?;
    let status = response.status();
    if !status.is_success() {
        bail!("DuckDuckGo returned HTTP {status}");
    }
    let html = response
        .text()
        .await
        .context("reading DuckDuckGo response")?;
    Ok(parse_duckduckgo(&html, max_results))
}

fn parse_duckduckgo(html: &str, max_results: usize) -> Vec<SearchResult> {
    // Compiled once: these run on every search, and regex compilation is far
    // more expensive than matching.
    static ANCHOR_RE: OnceLock<Regex> = OnceLock::new();
    static HREF_RE: OnceLock<Regex> = OnceLock::new();
    static SNIPPET_RE: OnceLock<Regex> = OnceLock::new();
    let anchor_re = ANCHOR_RE
        .get_or_init(|| Regex::new(r#"(?s)<a([^>]*class="result__a"[^>]*)>(.*?)</a>"#).unwrap());
    let href_re = HREF_RE.get_or_init(|| Regex::new(r#"href="([^"]+)""#).unwrap());
    let snippet_re = SNIPPET_RE.get_or_init(|| {
        Regex::new(r#"(?s)<(?:a|div)[^>]*class="result__snippet"[^>]*>(.*?)</(?:a|div)>"#).unwrap()
    });

    let mut titles: Vec<(String, String)> = Vec::new();
    for capture in anchor_re.captures_iter(html) {
        let attributes = capture.get(1).map(|m| m.as_str()).unwrap_or("");
        let title = capture.get(2).map(|m| m.as_str()).unwrap_or("");
        let Some(href) = href_re
            .captures(attributes)
            .and_then(|capture| capture.get(1))
        else {
            continue;
        };
        titles.push((clean_text(title), resolve_duckduckgo_url(href.as_str())));
    }

    let snippets: Vec<String> = snippet_re
        .captures_iter(html)
        .filter_map(|capture| capture.get(1).map(|m| clean_text(m.as_str())))
        .collect();

    titles
        .into_iter()
        .take(max_results)
        .enumerate()
        .map(|(index, (title, url))| SearchResult {
            title,
            url,
            snippet: snippets.get(index).cloned().unwrap_or_default(),
        })
        .collect()
}

async fn brave(
    query: &str,
    max_results: usize,
    api_key: Option<String>,
) -> Result<Vec<SearchResult>> {
    let key = api_key.context("web.api_key_env (BRAVE_API_KEY) is not set")?;
    let client = http_client()?;
    let url = format!(
        "https://api.search.brave.com/res/v1/web/search?q={}&count={}",
        percent_encode(query),
        max_results
    );
    let response = client
        .get(url)
        .header("X-Subscription-Token", key)
        .header("Accept", "application/json")
        .send()
        .await
        .context("Brave request")?;
    let status = response.status();
    if !status.is_success() {
        bail!("Brave returned HTTP {status}");
    }
    let value: Value = response.json().await.context("parsing Brave response")?;
    let mut results = Vec::new();
    if let Some(items) = value["web"]["results"].as_array() {
        for item in items.iter().take(max_results) {
            results.push(SearchResult {
                title: item["title"].as_str().unwrap_or("").to_string(),
                url: item["url"].as_str().unwrap_or("").to_string(),
                snippet: item["description"].as_str().unwrap_or("").to_string(),
            });
        }
    }
    Ok(results)
}

async fn tavily(
    query: &str,
    max_results: usize,
    api_key: Option<String>,
) -> Result<Vec<SearchResult>> {
    let key = api_key.context("web.api_key_env (TAVILY_API_KEY) is not set")?;
    let client = http_client()?;
    let body = json!({"api_key": key, "query": query, "max_results": max_results});
    let response = client
        .post("https://api.tavily.com/search")
        .json(&body)
        .send()
        .await
        .context("Tavily request")?;
    let status = response.status();
    if !status.is_success() {
        bail!("Tavily returned HTTP {status}");
    }
    let value: Value = response.json().await.context("parsing Tavily response")?;
    let mut results = Vec::new();
    if let Some(items) = value["results"].as_array() {
        for item in items.iter().take(max_results) {
            results.push(SearchResult {
                title: item["title"].as_str().unwrap_or("").to_string(),
                url: item["url"].as_str().unwrap_or("").to_string(),
                snippet: item["content"].as_str().unwrap_or("").to_string(),
            });
        }
    }
    Ok(results)
}

async fn searxng(
    query: &str,
    max_results: usize,
    base_url: Option<&str>,
) -> Result<Vec<SearchResult>> {
    let base = base_url.context("web.searxng_url is not set")?;
    let client = http_client()?;
    let url = format!(
        "{}/search?q={}&format=json",
        base.trim_end_matches('/'),
        percent_encode(query)
    );
    let response = client.get(url).send().await.context("SearXNG request")?;
    let status = response.status();
    if !status.is_success() {
        bail!("SearXNG returned HTTP {status}");
    }
    let value: Value = response.json().await.context("parsing SearXNG response")?;
    let mut results = Vec::new();
    if let Some(items) = value["results"].as_array() {
        for item in items.iter().take(max_results) {
            results.push(SearchResult {
                title: item["title"].as_str().unwrap_or("").to_string(),
                url: item["url"].as_str().unwrap_or("").to_string(),
                snippet: item["content"].as_str().unwrap_or("").to_string(),
            });
        }
    }
    Ok(results)
}

fn resolve_duckduckgo_url(href: &str) -> String {
    if let Some(position) = href.find("uddg=") {
        let rest = &href[position + 5..];
        let end = rest.find('&').unwrap_or(rest.len());
        return percent_decode(&rest[..end]);
    }
    if href.starts_with("//") {
        return format!("https:{href}");
    }
    href.to_string()
}

fn percent_encode(input: &str) -> String {
    let mut out = String::new();
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or("");
            if let Ok(value) = u8::from_str_radix(hex, 16) {
                out.push(value);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn clean_text(html: &str) -> String {
    collapse_whitespace(&decode_entities(&strip_html(html)))
}

fn strip_html(html: &str) -> String {
    static TAG_RE: OnceLock<Regex> = OnceLock::new();
    let tag_re = TAG_RE.get_or_init(|| Regex::new(r"(?s)<[^>]+>").unwrap());
    tag_re.replace_all(html, " ").into_owned()
}

fn decode_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&nbsp;", " ")
}

fn collapse_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.push(ch);
            last_space = false;
        }
    }
    out.trim().to_string()
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push_str("\n[truncated]");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_encoding_roundtrip() {
        let encoded = percent_encode("rust async runtime");
        assert_eq!(encoded, "rust+async+runtime");
        assert_eq!(percent_decode("a%20b"), "a b");
    }

    #[test]
    fn duckduckgo_redirect_is_resolved() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc";
        assert_eq!(resolve_duckduckgo_url(href), "https://example.com/page");
    }

    #[test]
    fn duckduckgo_html_is_parsed() {
        let html = r#"
        <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frust-lang.org&rut=x">Rust Programming Language</a>
        <a class="result__snippet" href="https://rust-lang.org">A language empowering everyone.</a>
        "#;
        let results = parse_duckduckgo(html, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Rust Programming Language");
        assert_eq!(results[0].url, "https://rust-lang.org");
        assert!(results[0].snippet.contains("empowering"));
    }

    #[test]
    fn html_is_stripped_and_entities_decoded() {
        let text = clean_text("<p>Hello &amp; <b>world</b></p>");
        assert_eq!(text, "Hello & world");
    }

    #[test]
    fn ssrf_guard_blocks_private_and_loopback_targets() {
        for url in [
            "http://127.0.0.1:11234/v1",
            "http://localhost/admin",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.4.108:11234/v1",
            "http://192.168.1.1/",
            "http://172.16.0.1/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://0.0.0.0/",
            "http://100.64.0.1/",
        ] {
            assert!(guard_url(url).is_err(), "{url} should be blocked");
        }
    }

    #[test]
    fn ssrf_guard_allows_public_hosts() {
        // Literal public IPs need no DNS, so these stay offline-safe.
        for url in ["https://8.8.8.8/", "http://93.184.216.34/"] {
            assert!(guard_url(url).is_ok(), "{url} should be allowed");
        }
    }

    #[test]
    fn blocked_ip_classification() {
        assert!(is_blocked_ip("127.0.0.1".parse().unwrap()));
        assert!(is_blocked_ip("169.254.169.254".parse().unwrap()));
        assert!(is_blocked_ip("10.0.0.1".parse().unwrap()));
        assert!(is_blocked_ip("::1".parse().unwrap()));
        assert!(is_blocked_ip("fe80::1".parse().unwrap()));
        assert!(is_blocked_ip("fc00::1".parse().unwrap()));
        assert!(!is_blocked_ip("8.8.8.8".parse().unwrap()));
        assert!(!is_blocked_ip("2606:4700::1111".parse().unwrap()));
    }

    #[test]
    fn fetch_rejects_non_http_urls() {
        let ctx = ToolContext::new(
            std::path::PathBuf::from("/tmp"),
            rustyclippy_core::AgentMode::Build,
        );
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result =
            runtime.block_on(WebFetchTool.execute(&json!({"url": "file:///etc/passwd"}), &ctx));
        assert!(result.is_err());
    }
}
