use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    Ident(String),
    Number(f64),
    StringLiteral(String),
    BoolLiteral(bool),

    // Keywords - Standard & Control Flow
    KeywordLet,
    KeywordFn,
    KeywordReturn,
    KeywordIf,
    KeywordElse,
    KeywordFor,
    KeywordIn,
    KeywordWhile,
    KeywordMut,
    KeywordLikh, // Print output keyword

    // Keywords - Math, Neural Net & Matrix Primitives
    KeywordMat,
    KeywordTensor,
    KeywordAttn,
    KeywordGrad,
    KeywordRequiresGrad,
    KeywordBackward,

    // Keywords - Visualization Suite
    KeywordPlot,
    KeywordFigure,
    KeywordHeatmap,
    KeywordShow,

    // Mathematical & Arithmetic Operators
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    Caret,      // ^ (Exponentiation)
    At,         // @ (Matrix Multiplication)
    SingleQuote,// ' (Transpose operator, e.g. A')

    // Comparison Operators
    EqualEqual, // ==
    NotEqual,   // !=
    Less,       // <
    LessEqual,  // <=
    Greater,    // >
    GreaterEqual,// >=

    // Assignment & Compound Assignment
    Equal,      // =
    PlusEqual,  // +=
    MinusEqual, // -=
    StarEqual,  // *=
    SlashEqual, // /=
    AtEqual,    // @=

    // Logical & Bitwise
    AmpAmp,     // &&
    PipePipe,   // ||
    Bang,       // !

    // Punctuation & Delimiters
    LParen,     // (
    RParen,     // )
    LBrace,     // {
    RBrace,     // }
    LBracket,   // [
    RBracket,   // ]
    Comma,      // ,
    Colon,      // :
    DoubleColon,// ::
    Semicolon,  // ;
    Dot,        // .
    DotDot,     // ..
    Arrow,      // ->
    FatArrow,   // =>

    // End of File / Unknown
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Ident(s) => write!(f, "Identifier('{}')", s),
            TokenKind::Number(n) => write!(f, "Number({})", n),
            TokenKind::StringLiteral(s) => write!(f, "String(\"{}\")", s),
            TokenKind::BoolLiteral(b) => write!(f, "Bool({})", b),
            TokenKind::KeywordLikh => write!(f, "Keyword('likh')"),
            TokenKind::Plus => write!(f, "'+'"),
            TokenKind::Minus => write!(f, "'-'"),
            TokenKind::Star => write!(f, "'*'"),
            TokenKind::Slash => write!(f, "'/'"),
            TokenKind::At => write!(f, "'@'"),
            TokenKind::Caret => write!(f, "'^'"),
            TokenKind::Equal => write!(f, "'='"),
            TokenKind::Eof => write!(f, "EOF"),
            _ => write!(f, "{:?}", self),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    pub line: usize,
    pub column: usize,
    pub index: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub position: Position,
}

pub struct Lexer<'a> {
    source: &'a str,
    chars: Vec<(usize, char)>,
    cursor: usize,
    line: usize,
    column: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        let chars: Vec<(usize, char)> = source.char_indices().collect();
        Self {
            source,
            chars,
            cursor: 0,
            line: 1,
            column: 1,
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.chars.get(self.cursor).map(|(_, c)| *c)
    }

    fn peek_next_char(&self) -> Option<char> {
        self.chars.get(self.cursor + 1).map(|(_, c)| *c)
    }

    fn advance(&mut self) -> Option<char> {
        if let Some((_, c)) = self.chars.get(self.cursor) {
            let ch = *c;
            self.cursor += 1;
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            Some(ch)
        } else {
            None
        }
    }

    fn get_position(&self) -> Position {
        Position {
            line: self.line,
            column: self.column,
            index: self.cursor,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();

        while let Some(ch) = self.peek_char() {
            let pos = self.get_position();

            match ch {
                // Whitespace
                ' ' | '\t' | '\r' | '\n' => {
                    self.advance();
                }

                // Comments: // or /* ... */
                '/' => {
                    if self.peek_next_char() == Some('/') {
                        // Single-line comment
                        self.advance(); // consume '/'
                        self.advance(); // consume '/'
                        while let Some(c) = self.peek_char() {
                            if c == '\n' {
                                break;
                            }
                            self.advance();
                        }
                    } else if self.peek_next_char() == Some('*') {
                        // Multi-line comment
                        self.advance(); // consume '/'
                        self.advance(); // consume '*'
                        let mut closed = false;
                        while let Some(c) = self.peek_char() {
                            if c == '*' && self.peek_next_char() == Some('/') {
                                self.advance(); // consume '*'
                                self.advance(); // consume '/'
                                closed = true;
                                break;
                            }
                            self.advance();
                        }
                        if !closed {
                            return Err(format!("Unterminated block comment at line {}, col {}", pos.line, pos.column));
                        }
                    } else if self.peek_next_char() == Some('=') {
                        self.advance();
                        self.advance();
                        tokens.push(Token { kind: TokenKind::SlashEqual, position: pos });
                    } else {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::Slash, position: pos });
                    }
                }

                // Numbers
                c if c.is_ascii_digit() => {
                    let num_token = self.read_number(pos)?;
                    tokens.push(num_token);
                }

                // String Literals
                '"' => {
                    let str_token = self.read_string(pos)?;
                    tokens.push(str_token);
                }

                // Identifiers & Keywords
                c if c.is_ascii_alphabetic() || c == '_' => {
                    let ident_token = self.read_identifier_or_keyword(pos);
                    tokens.push(ident_token);
                }

                // Multi-character & Single-character operators
                '+' => {
                    self.advance();
                    if self.peek_char() == Some('=') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::PlusEqual, position: pos });
                    } else {
                        tokens.push(Token { kind: TokenKind::Plus, position: pos });
                    }
                }
                '-' => {
                    self.advance();
                    if self.peek_char() == Some('>') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::Arrow, position: pos });
                    } else if self.peek_char() == Some('=') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::MinusEqual, position: pos });
                    } else {
                        tokens.push(Token { kind: TokenKind::Minus, position: pos });
                    }
                }
                '*' => {
                    self.advance();
                    if self.peek_char() == Some('=') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::StarEqual, position: pos });
                    } else {
                        tokens.push(Token { kind: TokenKind::Star, position: pos });
                    }
                }
                '%' => {
                    self.advance();
                    tokens.push(Token { kind: TokenKind::Percent, position: pos });
                }
                '^' => {
                    self.advance();
                    tokens.push(Token { kind: TokenKind::Caret, position: pos });
                }
                '@' => {
                    self.advance();
                    if self.peek_char() == Some('=') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::AtEqual, position: pos });
                    } else {
                        tokens.push(Token { kind: TokenKind::At, position: pos });
                    }
                }
                '\'' => {
                    self.advance();
                    tokens.push(Token { kind: TokenKind::SingleQuote, position: pos });
                }
                '=' => {
                    self.advance();
                    if self.peek_char() == Some('=') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::EqualEqual, position: pos });
                    } else if self.peek_char() == Some('>') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::FatArrow, position: pos });
                    } else {
                        tokens.push(Token { kind: TokenKind::Equal, position: pos });
                    }
                }
                '!' => {
                    self.advance();
                    if self.peek_char() == Some('=') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::NotEqual, position: pos });
                    } else {
                        tokens.push(Token { kind: TokenKind::Bang, position: pos });
                    }
                }
                '<' => {
                    self.advance();
                    if self.peek_char() == Some('=') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::LessEqual, position: pos });
                    } else {
                        tokens.push(Token { kind: TokenKind::Less, position: pos });
                    }
                }
                '>' => {
                    self.advance();
                    if self.peek_char() == Some('=') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::GreaterEqual, position: pos });
                    } else {
                        tokens.push(Token { kind: TokenKind::Greater, position: pos });
                    }
                }
                '&' => {
                    self.advance();
                    if self.peek_char() == Some('&') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::AmpAmp, position: pos });
                    } else {
                        return Err(format!("Unexpected character '&' at line {}, col {}", pos.line, pos.column));
                    }
                }
                '|' => {
                    self.advance();
                    if self.peek_char() == Some('|') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::PipePipe, position: pos });
                    } else {
                        return Err(format!("Unexpected character '|' at line {}, col {}", pos.line, pos.column));
                    }
                }
                '(' => { self.advance(); tokens.push(Token { kind: TokenKind::LParen, position: pos }); }
                ')' => { self.advance(); tokens.push(Token { kind: TokenKind::RParen, position: pos }); }
                '{' => { self.advance(); tokens.push(Token { kind: TokenKind::LBrace, position: pos }); }
                '}' => { self.advance(); tokens.push(Token { kind: TokenKind::RBrace, position: pos }); }
                '[' => { self.advance(); tokens.push(Token { kind: TokenKind::LBracket, position: pos }); }
                ']' => { self.advance(); tokens.push(Token { kind: TokenKind::RBracket, position: pos }); }
                ',' => { self.advance(); tokens.push(Token { kind: TokenKind::Comma, position: pos }); }
                ';' => { self.advance(); tokens.push(Token { kind: TokenKind::Semicolon, position: pos }); }
                ':' => {
                    self.advance();
                    if self.peek_char() == Some(':') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::DoubleColon, position: pos });
                    } else {
                        tokens.push(Token { kind: TokenKind::Colon, position: pos });
                    }
                }
                '.' => {
                    self.advance();
                    if self.peek_char() == Some('.') {
                        self.advance();
                        tokens.push(Token { kind: TokenKind::DotDot, position: pos });
                    } else {
                        tokens.push(Token { kind: TokenKind::Dot, position: pos });
                    }
                }

                _ => {
                    return Err(format!("Unexpected character '{}' at line {}, col {}", ch, pos.line, pos.column));
                }
            }
        }

        tokens.push(Token {
            kind: TokenKind::Eof,
            position: self.get_position(),
        });

        Ok(tokens)
    }

    fn read_number(&mut self, pos: Position) -> Result<Token, String> {
        let mut num_str = String::new();
        let mut has_dot = false;

        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() {
                num_str.push(c);
                self.advance();
            } else if c == '.' && !has_dot && self.peek_next_char().map_or(false, |next| next.is_ascii_digit()) {
                has_dot = true;
                num_str.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Parse exponent notation if present (e.g. 1e-4)
        if let Some(c) = self.peek_char() {
            if c == 'e' || c == 'E' {
                num_str.push(c);
                self.advance();
                if let Some(sign) = self.peek_char() {
                    if sign == '+' || sign == '-' {
                        num_str.push(sign);
                        self.advance();
                    }
                }
                while let Some(digit) = self.peek_char() {
                    if digit.is_ascii_digit() {
                        num_str.push(digit);
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }

        num_str
            .parse::<f64>()
            .map(|val| Token {
                kind: TokenKind::Number(val),
                position: pos,
            })
            .map_err(|e| format!("Invalid numeric literal '{}' at line {}, col {}: {}", num_str, pos.line, pos.column, e))
    }

    fn read_string(&mut self, pos: Position) -> Result<Token, String> {
        self.advance(); // Consume leading '"'
        let mut val = String::new();

        while let Some(c) = self.peek_char() {
            if c == '"' {
                self.advance(); // Consume trailing '"'
                return Ok(Token {
                    kind: TokenKind::StringLiteral(val),
                    position: pos,
                });
            } else if c == '\\' {
                self.advance(); // Consume '\'
                match self.advance() {
                    Some('n') => val.push('\n'),
                    Some('t') => val.push('\t'),
                    Some('r') => val.push('\r'),
                    Some('\\') => val.push('\\'),
                    Some('"') => val.push('"'),
                    Some(other) => val.push(other),
                    None => return Err(format!("Unterminated escape sequence in string at line {}", pos.line)),
                }
            } else {
                val.push(c);
                self.advance();
            }
        }

        Err(format!("Unterminated string literal starting at line {}, col {}", pos.line, pos.column))
    }

    fn read_identifier_or_keyword(&mut self, pos: Position) -> Token {
        let mut ident = String::new();
        while let Some(c) = self.peek_char() {
            if c.is_ascii_alphanumeric() || c == '_' {
                ident.push(c);
                self.advance();
            } else {
                break;
            }
        }

        let kind = match ident.as_str() {
            // Control Flow & Declaration
            "let" => TokenKind::KeywordLet,
            "fn" => TokenKind::KeywordFn,
            "return" => TokenKind::KeywordReturn,
            "if" => TokenKind::KeywordIf,
            "else" => TokenKind::KeywordElse,
            "for" => TokenKind::KeywordFor,
            "in" => TokenKind::KeywordIn,
            "while" => TokenKind::KeywordWhile,
            "mut" => TokenKind::KeywordMut,
            "true" => TokenKind::BoolLiteral(true),
            "false" => TokenKind::BoolLiteral(false),
            "likh" => TokenKind::KeywordLikh,

            // Math, NN & Matrix Keywords
            "mat" | "Matrix" => TokenKind::KeywordMat,
            "tensor" | "Tensor" => TokenKind::KeywordTensor,
            "attn" => TokenKind::KeywordAttn,
            "grad" => TokenKind::KeywordGrad,
            "requires_grad" => TokenKind::KeywordRequiresGrad,
            "backward" => TokenKind::KeywordBackward,

            // Plotting Keywords
            "plot" => TokenKind::KeywordPlot,
            "figure" => TokenKind::KeywordFigure,
            "heatmap" => TokenKind::KeywordHeatmap,
            "show" => TokenKind::KeywordShow,

            _ => TokenKind::Ident(ident),
        };

        Token { kind, position: pos }
    }
}

/// Convenience entry point to tokenize source code text into a Vector of Tokens.
pub fn lex(source: &str) -> Result<Vec<Token>, String> {
    let mut lexer = Lexer::new(source);
    lexer.tokenize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_matrix_ops_and_keywords() {
        let src = r#"
            // Test Matrix Multiplication and Attention syntax
            let A: Matrix = [[1.0, 2.0], [3.0, 4.0]];
            let B = A @ A';
            likh("Result of matrix multiplication:", B);
            let attn_out = attn::self_attention(Q, K, V);
        "#;

        let tokens = lex(src).expect("Lexing failed");
        let kinds: Vec<TokenKind> = tokens.into_iter().map(|t| t.kind).collect();

        assert!(kinds.contains(&TokenKind::KeywordLet));
        assert!(kinds.contains(&TokenKind::KeywordMat));
        assert!(kinds.contains(&TokenKind::At));
        assert!(kinds.contains(&TokenKind::SingleQuote));
        assert!(kinds.contains(&TokenKind::KeywordLikh));
        assert!(kinds.contains(&TokenKind::KeywordAttn));
        assert!(kinds.contains(&TokenKind::DoubleColon));
        assert_eq!(kinds.last(), Some(&TokenKind::Eof));
    }

    #[test]
    fn test_lexer_comments_and_strings() {
        let src = "likh(\"Hello \\n Phaadu!\"); /* Block comment */";
        let tokens = lex(src).expect("Lexing failed");

        assert_eq!(tokens[0].kind, TokenKind::KeywordLikh);
        assert_eq!(tokens[1].kind, TokenKind::LParen);
        assert_eq!(tokens[2].kind, TokenKind::StringLiteral("Hello \n Phaadu!".to_string()));
        assert_eq!(tokens[3].kind, TokenKind::RParen);
        assert_eq!(tokens[4].kind, TokenKind::Semicolon);
    }
}

