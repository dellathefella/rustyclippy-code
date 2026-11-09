//! Raw-mode prompt input with direct clipboard image paste.
//!
//! Terminals cannot deliver image bytes as text, so the REPL reads keys in raw
//! mode and treats Ctrl+V as "attach the clipboard image". Bracketed paste is
//! enabled so pasted text (including dragged image paths) arrives intact; an
//! empty bracketed paste is treated as an image paste attempt.

use std::io::{self, IsTerminal, Read, Write};

use anyhow::{bail, Context, Result};
use rustyclippy_core::ImageAttachment;

use crate::images;

pub enum InputOutcome {
    Line(String),
    Exit,
}

/// Read one prompt line. Ctrl+V attaches a clipboard image to `pending`.
#[cfg(unix)]
pub fn read_line(prompt: &str, pending: &mut Vec<ImageAttachment>) -> Result<InputOutcome> {
    if !io::stdin().is_terminal() {
        return read_line_plain(prompt);
    }
    let _guard = RawMode::enable()?;
    print!("\x1b[?2004h");
    print!("{prompt}");
    io::stdout().flush().ok();

    let mut stdin = io::stdin().lock();
    let mut buffer = String::new();
    let mut cursor = 0usize;

    loop {
        let Some(byte) = read_byte(&mut stdin)? else {
            print!("\r\n");
            return Ok(InputOutcome::Exit);
        };
        match byte {
            0x03 => {
                print!("^C\r\n");
                return Ok(InputOutcome::Exit);
            }
            0x04 => {
                if buffer.is_empty() {
                    print!("\r\n");
                    return Ok(InputOutcome::Exit);
                }
            }
            0x0d | 0x0a => {
                print!("\r\n");
                return Ok(InputOutcome::Line(buffer));
            }
            0x16 => paste_clipboard(prompt, &buffer, cursor, pending)?,
            0x01 => {
                cursor = 0;
                redraw(prompt, &buffer, cursor);
            }
            0x05 => {
                cursor = buffer.len();
                redraw(prompt, &buffer, cursor);
            }
            0x0b => {
                buffer.truncate(cursor);
                redraw(prompt, &buffer, cursor);
            }
            0x15 => {
                buffer.clear();
                cursor = 0;
                redraw(prompt, &buffer, cursor);
            }
            0x17 => {
                delete_word(&mut buffer, &mut cursor);
                redraw(prompt, &buffer, cursor);
            }
            0x7f | 0x08 => {
                backspace(&mut buffer, &mut cursor);
                redraw(prompt, &buffer, cursor);
            }
            0x1b => handle_escape(&mut stdin, prompt, &mut buffer, &mut cursor, pending)?,
            byte if byte >= 0x20 => {
                let ch = read_char(&mut stdin, byte)?;
                insert_char(&mut buffer, &mut cursor, ch);
                redraw(prompt, &buffer, cursor);
            }
            _ => {}
        }
    }
}

/// Fallback for platforms without termios raw mode: Ctrl+V is unavailable, but
/// `/paste` and `/image` still work.
#[cfg(not(unix))]
pub fn read_line(prompt: &str, _pending: &mut Vec<ImageAttachment>) -> Result<InputOutcome> {
    read_line_plain(prompt)
}

fn read_line_plain(prompt: &str) -> Result<InputOutcome> {
    print!("{prompt}");
    io::stdout().flush().ok();
    let mut line = String::new();
    if io::stdin().read_line(&mut line)? == 0 {
        return Ok(InputOutcome::Exit);
    }
    Ok(InputOutcome::Line(
        line.trim_end_matches(['\r', '\n']).to_string(),
    ))
}

#[cfg(unix)]
struct RawMode {
    original: libc::termios,
}

#[cfg(unix)]
impl RawMode {
    fn enable() -> Result<Self> {
        let mut termios: libc::termios = unsafe { std::mem::zeroed() };
        if unsafe { libc::tcgetattr(libc::STDIN_FILENO, &mut termios) } != 0 {
            bail!("stdin is not a terminal; use `run` for non-interactive input");
        }
        let original = termios;
        unsafe { libc::cfmakeraw(&mut termios) };
        if unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSAFLUSH, &termios) } != 0 {
            bail!("failed to enable raw terminal mode");
        }
        Ok(Self { original })
    }
}

#[cfg(unix)]
impl Drop for RawMode {
    fn drop(&mut self) {
        unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSAFLUSH, &self.original) };
        print!("\x1b[?2004l");
        io::stdout().flush().ok();
    }
}

fn read_byte<R: Read>(stdin: &mut R) -> Result<Option<u8>> {
    let mut byte = [0u8; 1];
    match stdin.read(&mut byte) {
        Ok(0) => Ok(None),
        Ok(_) => Ok(Some(byte[0])),
        Err(error) => Err(error).context("reading stdin"),
    }
}

fn read_char<R: Read>(stdin: &mut R, first: u8) -> Result<char> {
    let mut bytes = vec![first];
    for _ in 1..utf8_width(first) {
        let Some(byte) = read_byte(stdin)? else { break };
        bytes.push(byte);
    }
    Ok(std::str::from_utf8(&bytes)
        .ok()
        .and_then(|text| text.chars().next())
        .unwrap_or('\u{FFFD}'))
}

fn utf8_width(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else if first >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

fn redraw(prompt: &str, buffer: &str, cursor: usize) {
    print!("\r{prompt}{buffer}\x1b[K");
    let tail = buffer[cursor..].chars().count();
    if tail > 0 {
        print!("\x1b[{tail}D");
    }
    io::stdout().flush().ok();
}

fn insert_char(buffer: &mut String, cursor: &mut usize, ch: char) {
    buffer.insert(*cursor, ch);
    *cursor += ch.len_utf8();
}

fn backspace(buffer: &mut String, cursor: &mut usize) {
    if *cursor == 0 {
        return;
    }
    let previous = buffer[..*cursor]
        .char_indices()
        .last()
        .map(|(index, _)| index)
        .unwrap_or(0);
    buffer.remove(previous);
    *cursor = previous;
}

fn delete_at(buffer: &mut String, cursor: usize) {
    if cursor < buffer.len() {
        buffer.remove(cursor);
    }
}

fn delete_word(buffer: &mut String, cursor: &mut usize) {
    let mut end = *cursor;
    while end > 0 {
        let previous = buffer[..end]
            .char_indices()
            .last()
            .map(|(index, _)| index)
            .unwrap_or(0);
        if !buffer[previous..end]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
        {
            break;
        }
        buffer.remove(previous);
        end = previous;
    }
    while end > 0 {
        let previous = buffer[..end]
            .char_indices()
            .last()
            .map(|(index, _)| index)
            .unwrap_or(0);
        if buffer[previous..end]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
        {
            break;
        }
        buffer.remove(previous);
        end = previous;
    }
    *cursor = end;
}

fn paste_clipboard(
    prompt: &str,
    buffer: &str,
    cursor: usize,
    pending: &mut Vec<ImageAttachment>,
) -> Result<()> {
    match images::capture_clipboard() {
        Ok(Some(image)) => {
            pending.push(image);
            print!("\r\n");
            println!("image attached · {} pending", pending.len());
        }
        Ok(None) => {
            print!("\r\n");
            println!("clipboard has no image");
        }
        Err(error) => {
            print!("\r\n");
            println!("clipboard capture failed: {error}");
        }
    }
    print!("{prompt}{buffer}");
    let tail = buffer[cursor..].chars().count();
    if tail > 0 {
        print!("\x1b[{tail}D");
    }
    io::stdout().flush().ok();
    Ok(())
}

fn handle_escape<R: Read>(
    stdin: &mut R,
    prompt: &str,
    buffer: &mut String,
    cursor: &mut usize,
    pending: &mut Vec<ImageAttachment>,
) -> Result<()> {
    let Some(next) = read_byte(stdin)? else {
        return Ok(());
    };
    if next != b'[' {
        return Ok(());
    }

    let mut sequence = Vec::new();
    loop {
        let Some(byte) = read_byte(stdin)? else {
            return Ok(());
        };
        sequence.push(byte);
        if (0x40..=0x7e).contains(&byte) {
            break;
        }
    }

    match sequence.as_slice() {
        b"A" => {} // no history yet
        b"B" => {} // no history yet
        b"C" => {
            if *cursor < buffer.len() {
                *cursor += buffer[*cursor..]
                    .chars()
                    .next()
                    .map(char::len_utf8)
                    .unwrap_or(1);
            }
            redraw(prompt, buffer, *cursor);
        }
        b"D" => {
            if *cursor > 0 {
                *cursor = buffer[..*cursor]
                    .char_indices()
                    .last()
                    .map(|(index, _)| index)
                    .unwrap_or(0);
            }
            redraw(prompt, buffer, *cursor);
        }
        b"H" | b"1~" => {
            *cursor = 0;
            redraw(prompt, buffer, *cursor);
        }
        b"F" | b"4~" => {
            *cursor = buffer.len();
            redraw(prompt, buffer, *cursor);
        }
        b"3~" => {
            delete_at(buffer, *cursor);
            redraw(prompt, buffer, *cursor);
        }
        b"200~" => {
            let mut paste = Vec::new();
            let mut tail = Vec::new();
            loop {
                let Some(byte) = read_byte(stdin)? else { break };
                paste.push(byte);
                tail.push(byte);
                if tail.len() > 6 {
                    tail.remove(0);
                }
                if tail.as_slice() == b"\x1b[201~" {
                    break;
                }
            }
            if paste.ends_with(b"\x1b[201~") {
                paste.truncate(paste.len() - 6);
            }
            let text = String::from_utf8_lossy(&paste).replace(['\r', '\n'], " ");
            if text.trim().is_empty() {
                paste_clipboard(prompt, buffer, *cursor, pending)?;
            } else {
                buffer.insert_str(*cursor, &text);
                *cursor += text.len();
                redraw(prompt, buffer, *cursor);
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delete_word_removes_trailing_word_and_spaces() {
        let mut buffer = String::from("explain this code   ");
        let mut cursor = buffer.len();
        delete_word(&mut buffer, &mut cursor);
        assert_eq!(buffer, "explain this ");
        assert_eq!(cursor, buffer.len());
    }

    #[test]
    fn delete_word_on_empty_buffer_is_noop() {
        let mut buffer = String::new();
        let mut cursor = 0;
        delete_word(&mut buffer, &mut cursor);
        assert!(buffer.is_empty());
        assert_eq!(cursor, 0);
    }

    #[test]
    fn utf8_width_matches_leading_byte() {
        assert_eq!(utf8_width(b'a'), 1);
        assert_eq!(utf8_width(0b1100_0010), 2);
        assert_eq!(utf8_width(0b1110_0010), 3);
        assert_eq!(utf8_width(0b1111_0000), 4);
    }

    #[test]
    fn bracketed_paste_inserts_text_at_cursor() {
        let mut input = std::io::Cursor::new(b"[200~hello world\x1b[201~".to_vec());
        let mut buffer = String::from("prefix ");
        let mut cursor = buffer.len();
        let mut pending = Vec::new();
        handle_escape(&mut input, "> ", &mut buffer, &mut cursor, &mut pending).unwrap();
        assert_eq!(buffer, "prefix hello world");
        assert_eq!(cursor, buffer.len());
        assert!(pending.is_empty());
    }

    #[test]
    fn arrow_sequence_moves_cursor() {
        let mut input = std::io::Cursor::new(b"[D".to_vec());
        let mut buffer = String::from("ab");
        let mut cursor = buffer.len();
        let mut pending = Vec::new();
        handle_escape(&mut input, "> ", &mut buffer, &mut cursor, &mut pending).unwrap();
        assert_eq!(cursor, 1);
    }

    #[test]
    fn backspace_removes_previous_char() {
        let mut buffer = String::from("abc");
        let mut cursor = buffer.len();
        backspace(&mut buffer, &mut cursor);
        assert_eq!(buffer, "ab");
        assert_eq!(cursor, 2);
    }

    #[test]
    fn insert_char_advances_cursor() {
        let mut buffer = String::from("ac");
        let mut cursor = 1;
        insert_char(&mut buffer, &mut cursor, 'b');
        assert_eq!(buffer, "abc");
        assert_eq!(cursor, 2);
    }
}
