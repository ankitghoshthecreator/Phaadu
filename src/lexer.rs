#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Ident(String),      // user-defined names like A, B, etc.
    Number(f64),        // numbers like 3, 4.5
    StringLiteral(String), // string values like "Hello"
    KeywordLikh,        // keyword for print
    Plus,
    Minus,
    Star,
    Slash,
    Equal,
    LParen,
    RParen,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub position: usize,
}

pub fn lex(source: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = source.chars().enumerate().peekable();

    while let Some((i, ch)) = chars.next() {
        match ch {
            // ignore whitespace
            ' ' | '\n' | '\t' => continue,

            // single-character tokens
            '+' => tokens.push(Token { kind: TokenKind::Plus, position: i }),
            '-' => tokens.push(Token { kind: TokenKind::Minus, position: i }),
            '*' => tokens.push(Token { kind: TokenKind::Star, position: i }),
            '/' => tokens.push(Token { kind: TokenKind::Slash, position: i }),
            '=' => tokens.push(Token { kind: TokenKind::Equal, position: i }),
            '(' => tokens.push(Token { kind: TokenKind::LParen, position: i }),
            ')' => tokens.push(Token { kind: TokenKind::RParen, position: i }),

            // number tokens
            c if c.is_ascii_digit() => {
                let mut num = c.to_string();
                while let Some((_, next)) = chars.peek() {
                    if next.is_ascii_digit() || *next == '.' {
                        num.push(*next);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token {
                    kind: TokenKind::Number(num.parse().unwrap()),
                    position: i,
                });
            }

            // string literal tokens
            '"' => {
                let mut string_val = String::new();
                while let Some((_, next)) = chars.next() {
                    if next == '"' {
                        break;
                    }
                    string_val.push(next);
                }
                tokens.push(Token {
                    kind: TokenKind::StringLiteral(string_val),
                    position: i,
                });
            }

            // identifiers or keywords
            c if c.is_ascii_alphabetic() => {
                let mut name = c.to_string();
                while let Some((_, next)) = chars.peek() {
                    if next.is_ascii_alphanumeric() {
                        name.push(*next);
                        chars.next();
                    } else {
                        break;
                    }
                }

                // check for keyword
                let kind = match name.as_str() {
                    "likh" => TokenKind::KeywordLikh,
                    _ => TokenKind::Ident(name),
                };

                tokens.push(Token { kind, position: i });
            }

            // anything else is invalid
            _ => return Err(format!("Unexpected character '{}' at {}", ch, i)),
        }
    }

    Ok(tokens)
}
