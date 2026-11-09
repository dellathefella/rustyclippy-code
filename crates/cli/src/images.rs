//! Image acquisition: clipboard capture and file loading.
//!
//! Terminals cannot paste image bytes as text, so the REPL offers `/paste`
//! (read the clipboard image through the platform's native tooling) and
//! `/image <path>` (attach a file). Both produce an [`ImageAttachment`] that
//! travels in the next user message.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use rustyclippy_core::{detect_mime, ImageAttachment};

/// Refuse absurd payloads before they bloat the session log and the prompt.
pub const MAX_IMAGE_BYTES: usize = 16 * 1024 * 1024;

pub fn read_image(path: &Path) -> Result<ImageAttachment> {
    let bytes = std::fs::read(path).with_context(|| format!("reading image {}", path.display()))?;
    attachment_from_bytes(&bytes, &path.display().to_string())
}

fn attachment_from_bytes(bytes: &[u8], source: &str) -> Result<ImageAttachment> {
    if bytes.is_empty() {
        bail!("{source} is empty");
    }
    if bytes.len() > MAX_IMAGE_BYTES {
        bail!(
            "{source} is {} bytes; the limit is {} bytes",
            bytes.len(),
            MAX_IMAGE_BYTES
        );
    }
    let Some(mime) = detect_mime(bytes) else {
        bail!("{source} is not a PNG, JPEG, GIF, or WebP image");
    };
    Ok(ImageAttachment::from_bytes(mime, bytes))
}

/// Capture an image from the system clipboard. Returns `Ok(None)` when the
/// clipboard holds no image (or no capture tool is available).
pub fn capture_clipboard() -> Result<Option<ImageAttachment>> {
    #[cfg(target_os = "macos")]
    let bytes = capture_macos()?;
    #[cfg(target_os = "linux")]
    let bytes = capture_linux()?;
    #[cfg(target_os = "windows")]
    let bytes = capture_windows()?;
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let bytes: Option<Vec<u8>> = None;

    match bytes {
        Some(bytes) => Ok(Some(attachment_from_bytes(&bytes, "clipboard image")?)),
        None => Ok(None),
    }
}

fn temp_path(extension: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "rustyclippy-clip-{}-{nanos}.{extension}",
        std::process::id()
    ))
}

fn run(command: &mut Command) -> Option<Vec<u8>> {
    let output = command.output().ok()?;
    if output.status.success() && !output.stdout.is_empty() {
        Some(output.stdout)
    } else {
        None
    }
}

/// Run a command and report success without leaking its stderr into the REPL.
/// macOS ImageIO prints JP2 color-space warnings for some clipboard images.
fn succeeded(command: &mut Command) -> bool {
    matches!(command.output(), Ok(output) if output.status.success())
}

#[cfg(target_os = "macos")]
fn capture_macos() -> Result<Option<Vec<u8>>> {
    // 1. `pngpaste` is the cleanest path when installed.
    if let Some(bytes) = run(Command::new("pngpaste").arg("-")) {
        if detect_mime(&bytes).is_some() {
            return Ok(Some(bytes));
        }
    }

    // 2. AppleScript: write the clipboard PNG representation to a temp file.
    let png = temp_path("png");
    let script = format!(
        "set pngData to the clipboard as «class PNGf»\n\
         set outFile to open for access POSIX file \"{}\" with write permission\n\
         write pngData to outFile\n\
         close access outFile",
        png.display()
    );
    let mut command = Command::new("osascript");
    command.arg("-e").arg(&script);
    if succeeded(&mut command) {
        if let Ok(bytes) = std::fs::read(&png) {
            let _ = std::fs::remove_file(&png);
            if detect_mime(&bytes).is_some() {
                return Ok(Some(bytes));
            }
        }
    }
    let _ = std::fs::remove_file(&png);

    // 3. TIFF fallback: macOS often stores clipboard images as TIFF. Convert
    //    with `sips`.
    let tiff = temp_path("tiff");
    let script = format!(
        "set tiffData to the clipboard as «class TIFF»\n\
         set outFile to open for access POSIX file \"{}\" with write permission\n\
         write tiffData to outFile\n\
         close access outFile",
        tiff.display()
    );
    let mut command = Command::new("osascript");
    command.arg("-e").arg(&script);
    if succeeded(&mut command) {
        let converted = temp_path("png");
        let mut sips = Command::new("sips");
        sips.args(["-s", "format", "png"])
            .arg(&tiff)
            .arg("--out")
            .arg(&converted);
        let bytes = if succeeded(&mut sips) {
            std::fs::read(&converted).ok()
        } else {
            None
        };
        let _ = std::fs::remove_file(&tiff);
        let _ = std::fs::remove_file(&converted);
        if let Some(bytes) = bytes {
            if detect_mime(&bytes).is_some() {
                return Ok(Some(bytes));
            }
        }
    }
    let _ = std::fs::remove_file(&tiff);

    Ok(None)
}

#[cfg(target_os = "linux")]
fn capture_linux() -> Result<Option<Vec<u8>>> {
    let mut commands: Vec<Vec<&str>> = Vec::new();
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        commands.push(vec!["wl-paste", "--type", "image/png", "--no-newline"]);
    }
    commands.push(vec![
        "xclip",
        "-selection",
        "clipboard",
        "-t",
        "image/png",
        "-o",
    ]);
    commands.push(vec![
        "xclip",
        "-selection",
        "clipboard",
        "-t",
        "image/jpeg",
        "-o",
    ]);

    for parts in commands {
        let mut command = Command::new(parts[0]);
        command.args(&parts[1..]);
        if let Some(bytes) = run(&mut command) {
            if detect_mime(&bytes).is_some() {
                return Ok(Some(bytes));
            }
        }
    }
    Ok(None)
}

#[cfg(target_os = "windows")]
fn capture_windows() -> Result<Option<Vec<u8>>> {
    let png = temp_path("png");
    let script = format!(
        "Add-Type -AssemblyName System.Windows.Forms,System.Drawing\n\
         $img = [System.Windows.Forms.Clipboard]::GetImage()\n\
         if ($null -eq $img) {{ exit 2 }}\n\
         $img.Save(\"{}\", [System.Drawing.Imaging.ImageFormat]::Png)",
        png.display()
    );
    let mut command = Command::new("powershell");
    command.args(["-NoProfile", "-Command", &script]);
    if succeeded(&mut command) {
        if let Ok(bytes) = std::fs::read(&png) {
            let _ = std::fs::remove_file(&png);
            if detect_mime(&bytes).is_some() {
                return Ok(Some(bytes));
            }
        }
    }
    let _ = std::fs::remove_file(&png);
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_images() {
        let error = attachment_from_bytes(b"hello", "test").unwrap_err();
        assert!(error.to_string().contains("not a PNG"));
    }

    #[test]
    fn accepts_png_magic() {
        let bytes = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0];
        let image = attachment_from_bytes(&bytes, "test").unwrap();
        assert_eq!(image.mime, "image/png");
    }

    #[test]
    fn rejects_oversized_images() {
        let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        bytes.resize(MAX_IMAGE_BYTES + 1, 0);
        assert!(attachment_from_bytes(&bytes, "test").is_err());
    }
}
