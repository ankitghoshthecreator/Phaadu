use crate::lexer::{Token, TokenKind};

#[derive(Debug)]
pub enum AstNode {
    PrintCall(String), // likh("text")
}

pub fn parse(tokens: &[Token]) -> Result<AstNode, String> {
    let mut pos = 0;

    // helper to peek safely
    let peek = |i: usize| tokens.get(i).map(|t| &t.kind);

    match peek(pos) {
        Some(TokenKind::KeywordLikh) => {
            pos += 1;
            // expect '('
            match peek(pos) {
                Some(TokenKind::LParen) => pos += 1,
                _ => return Err("Expected '(' after 'likh'".into()),
            }

            // expect string literal
            let arg = match peek(pos) {
                Some(TokenKind::StringLiteral(s)) => {
                    pos += 1;
                    s.clone()
                }
                _ => return Err("Expected string after '('".into()),
            };

            // expect ')'
            match peek(pos) {
                Some(TokenKind::RParen) => (),
                _ => return Err("Expected ')' after string".into()),
            }

            Ok(AstNode::PrintCall(arg))
        }
        _ => Err("Expected 'likh' statement".into()),
    }
}
