//! Interactive setup wizard.
//!
//! Detects an existing DSH harness configuration, probes common local
//! OpenAI-compatible endpoints, and writes a validated `deeprust.json`.

use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Result};
use deeprust_core::{
    session::default_home, Config, PermissionConfig, PermissionMode, ProviderConfig, ProviderKind,
};

const CANDIDATES: &[&str] = &[
    "http://127.0.0.1:8080/v1",
    "http://127.0.0.1:11234/v1",
    "http://localhost:11434/v1",
    "http://127.0.0.1:1234/v1",
    "http://127.0.0.1:8000/v1",
];

struct Detection {
    source: String,
    base_url: Option<String>,
    model: Option<String>,
    reasoning_effort: Option<String>,
}

pub async fn run(path: Option<PathBuf>, yes: bool, force: bool) -> Result<()> {
    let target = path.unwrap_or_else(|| default_home().join("config.json"));
    if target.exists() && !force {
        bail!(
            "{} already exists; pass --force to overwrite",
            target.display()
        );
    }

    println!("deeprust setup");
    println!("──────────────");

    let interactive = !yes && io::stdin().is_terminal();

    // 1. Reuse the DSH harness configuration when present.
    let detected = detect_dsh_settings();
    let mut base_url: Option<String> = None;
    let mut model: Option<String> = None;
    let mut effort: Option<String> = None;
    let mut kind = ProviderKind::OpenAiCompat;
    let mut permissions = PermissionConfig::default();

    if let Some(detection) = &detected {
        println!("found DSH harness settings ({})", detection.source);
        if let Some(url) = &detection.base_url {
            println!("  endpoint:         {url}");
            base_url = Some(url.clone());
        }
        if let Some(name) = &detection.model {
            println!("  model:            {name}");
            model = Some(name.clone());
        }
        if let Some(value) = &detection.reasoning_effort {
            println!("  reasoning effort: {value}");
            effort = Some(value.clone());
        }
    }

    // 2. Probe local endpoints when nothing was detected.
    if base_url.is_none() {
        println!("probing local endpoints...");
        for candidate in CANDIDATES {
            if let Some(models) = probe(candidate).await {
                println!("  {candidate} reachable ({} model(s))", models.len());
                base_url = Some(candidate.to_string());
                if model.is_none() {
                    model = models.first().cloned();
                }
                break;
            }
        }
    }

    // 3. Ask the user.
    if interactive {
        println!();
        println!("provider:");
        println!("  1. local OpenAI-compatible (MLX / llama.cpp / vLLM / LM Studio / Ollama)");
        println!("  2. DeepSeek hosted API");
        let provider_choice = prompt("provider", "1")?;
        if provider_choice.trim() == "2" {
            kind = ProviderKind::Deepseek;
            base_url = Some("https://api.deepseek.com/v1".to_string());
            model = Some("deepseek-v4.1-flash".to_string());
            effort = None;
        } else {
            let default_url = base_url
                .clone()
                .unwrap_or_else(|| "http://127.0.0.1:8080/v1".to_string());
            base_url = Some(prompt("endpoint base URL", &default_url)?);
            let endpoint = base_url.clone().unwrap();
            let default_model = model
                .clone()
                .unwrap_or_else(|| ProviderConfig::default().model);
            if let Some(models) = probe(&endpoint).await {
                if !models.is_empty() {
                    println!("available models:");
                    for (index, name) in models.iter().enumerate() {
                        println!("  {}. {name}", index + 1);
                    }
                    let answer = prompt("model (number or id)", &default_model)?;
                    model = Some(match answer.parse::<usize>() {
                        Ok(index) if index >= 1 && index <= models.len() => {
                            models[index - 1].clone()
                        }
                        _ => answer,
                    });
                } else {
                    model = Some(prompt("model id", &default_model)?);
                }
            } else {
                println!("endpoint did not answer /models; enter the model id manually");
                model = Some(prompt("model id", &default_model)?);
            }
        }

        println!("reasoning effort:");
        println!("  1. off");
        println!("  2. low");
        println!("  3. high");
        println!("  4. max");
        let default_effort = effort.clone().unwrap_or_else(|| "high".to_string());
        let answer = prompt("reasoning effort", &default_effort)?;
        effort = Some(match answer.trim().to_lowercase().as_str() {
            "1" | "off" => "off".to_string(),
            "2" | "low" => "low".to_string(),
            "4" | "max" => "max".to_string(),
            _ => "high".to_string(),
        });

        println!("permissions for mutating tools:");
        println!("  1. ask (prompt before bash/edit/write/task)");
        println!("  2. auto (run without prompting)");
        println!("  3. readonly (deny mutating tools)");
        let permission = parse_permission(&prompt("permission mode", "1")?);
        permissions = PermissionConfig {
            bash: permission,
            edit: permission,
            write: permission,
            task: permission,
        };
    } else {
        model = model.or_else(|| Some(ProviderConfig::default().model));
        effort = effort.or_else(|| Some("high".to_string()));
        if base_url
            .as_deref()
            .map(|url| url.contains("api.deepseek.com"))
            .unwrap_or(false)
        {
            kind = ProviderKind::Deepseek;
        }
    }

    let context_tokens = 262_144usize;
    let autocompact = 250_000usize;

    let config = Config {
        provider: ProviderConfig {
            kind,
            base_url: base_url
                .clone()
                .unwrap_or_else(|| "http://127.0.0.1:8080/v1".to_string()),
            model: model.unwrap_or_else(|| ProviderConfig::default().model),
            max_context_tokens: context_tokens,
            max_output_tokens: 8_192,
            reasoning_effort: effort,
            cache_prompt: kind == ProviderKind::OpenAiCompat,
            api_key_env: if kind == ProviderKind::Deepseek {
                "DEEPSEEK_API_KEY".to_string()
            } else {
                String::new()
            },
            ..ProviderConfig::default()
        },
        cache: deeprust_core::CacheConfig {
            autocompact_at_tokens: autocompact,
            ..deeprust_core::CacheConfig::default()
        },
        permissions,
        ..Config::default()
    };
    config.validate()?;
    config.save(&target)?;

    println!();
    println!("wrote {}", target.display());
    println!("  endpoint:         {}", config.provider.base_url);
    println!("  model:            {}", config.provider.model);
    println!(
        "  reasoning effort: {}",
        config
            .provider
            .effective_reasoning_effort()
            .unwrap_or("off")
    );
    println!(
        "  autocompact at:   {} tokens",
        config.cache.autocompact_at_tokens
    );
    println!();
    println!("next: `deeprust doctor` to verify the endpoint, then `deeprust chat`.");
    Ok(())
}

fn parse_permission(value: &str) -> PermissionMode {
    match value.trim().to_lowercase().as_str() {
        "2" | "auto" => PermissionMode::Auto,
        "3" | "readonly" => PermissionMode::Readonly,
        _ => PermissionMode::Ask,
    }
}

fn prompt(question: &str, default: &str) -> Result<String> {
    print!("{question} [{default}]: ");
    let _ = io::stdout().flush();
    let mut line = String::new();
    if io::stdin().read_line(&mut line)? == 0 {
        return Ok(default.to_string());
    }
    let trimmed = line.trim();
    if trimmed.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

async fn probe(base_url: &str) -> Option<Vec<String>> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(5))
        .build()
        .ok()?;
    let response = client.get(url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let value: serde_json::Value = response.json().await.ok()?;
    let models = value
        .get("data")?
        .as_array()?
        .iter()
        .filter_map(|model| model.get("id")?.as_str().map(|id| id.to_string()))
        .collect();
    Some(models)
}

/// Read the DSH harness settings if this machine has one.
fn detect_dsh_settings() -> Option<Detection> {
    let home = dirs::home_dir()?;
    let path = home.join(".dsh/vscode/harness-home/settings.yaml");
    let text = std::fs::read_to_string(&path).ok()?;

    let mut base_url = None;
    let mut model = None;
    let mut effort = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("baseURL:") {
            let value = rest.trim().trim_matches('"').trim_matches('\'');
            if !value.is_empty() {
                base_url = Some(value.to_string());
            }
        } else if let Some(rest) = trimmed.strip_prefix("model:") {
            let value = rest.trim().trim_matches('"').trim_matches('\'');
            if !value.is_empty() && model.is_none() {
                model = Some(value.to_string());
            }
        } else if let Some(rest) = trimmed.strip_prefix("reasoningEffort:") {
            let value = rest.trim().trim_matches('"').trim_matches('\'');
            if !value.is_empty() {
                effort = Some(value.to_string());
            }
        }
    }

    if base_url.is_none() && model.is_none() {
        return None;
    }
    Some(Detection {
        source: path.display().to_string(),
        base_url,
        model,
        reasoning_effort: effort,
    })
}

#[allow(dead_code)]
fn target_path(path: Option<PathBuf>) -> PathBuf {
    path.unwrap_or_else(|| default_home().join("config.json"))
}

#[allow(dead_code)]
fn exists(path: &Path) -> bool {
    path.exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_parsing() {
        assert_eq!(parse_permission("2"), PermissionMode::Auto);
        assert_eq!(parse_permission("readonly"), PermissionMode::Readonly);
        assert_eq!(parse_permission("1"), PermissionMode::Ask);
        assert_eq!(parse_permission("nonsense"), PermissionMode::Ask);
    }
}
