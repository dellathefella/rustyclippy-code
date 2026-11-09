//! Web tools: `websearch` (DuckDuckGo / Brave / Tavily / SearXNG) and `webfetch`.

use std::time::Duration;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use deeprust_core::config::WebProvider;
use regex::Regex;
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

fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60))
        .user_agent("Mozilla/5.0 (compatible; deeprust/0.1)")
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
    let anchor_re = Regex::new(r#"(?s)<a([^>]*class="result__a"[^>]*)>(.*?)</a>"#).unwrap();
    let href_re = Regex::new(r#"href="([^"]+)""#).unwrap();
    let snippet_re =
        Regex::new(r#"(?s)<(?:a|div)[^>]*class="result__snippet"[^>]*>(.*?)</(?:a|div)>"#).unwrap();

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
    let tag_re = Regex::new(r"(?s)<[^>]+>").unwrap();
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
    fn fetch_rejects_non_http_urls() {
        let ctx = ToolContext::new(
            std::path::PathBuf::from("/tmp"),
            deeprust_core::AgentMode::Build,
        );
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result =
            runtime.block_on(WebFetchTool.execute(&json!({"url": "file:///etc/passwd"}), &ctx));
        assert!(result.is_err());
    }
}
