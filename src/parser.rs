use crate::lexer::{Position, Token, TokenKind};
use std::fmt;

// ============================================================================
// 1. AST Data Structures
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOpKind {
    Add,        // +
    Sub,        // -
    Mul,        // *
    Div,        // /
    Mod,        // %
    Pow,        // ^ (exponentiation)
    MatMul,     // @ (matrix multiplication)
    Eq,         // ==
    Neq,        // !=
    Lt,         // <
    Lte,        // <=
    Gt,         // >
    Gte,        // >=
    And,        // &&
    Or,         // ||
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOpKind {
    Neg,        // -
    Not,        // !
    Transpose,  // ' (Postfix matrix transpose)
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    F64,
    Bool,
    String,
    Matrix { rows: Option<usize>, cols: Option<usize> },
    Tensor { dimensions: Vec<usize> },
    Custom(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    StringLiteral(String),
    BoolLiteral(bool),
    Identifier(String),
    MatrixLiteral(Vec<Vec<Expr>>),
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOpKind,
        right: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOpKind,
        expr: Box<Expr>,
    },
    FunctionCall {
        callee: String,
        args: Vec<Expr>,
    },
    MemberAccess {
        object: Box<Expr>,
        member: String,
    },
    Range {
        start: Box<Expr>,
        end: Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    VarDecl {
        name: String,
        is_mutable: bool,
        type_annotation: Option<TypeKind>,
        initializer: Expr,
    },
    Assignment {
        target: String,
        op: Option<BinaryOpKind>, // None for '=' or Some(MatMul) for '@='
        value: Expr,
    },
    PrintStmt {
        args: Vec<Expr>,
    },
    BackwardStmt {
        target: Expr,
    },
    IfStmt {
        condition: Expr,
        then_branch: Vec<Statement>,
        else_branch: Option<Vec<Statement>>,
    },
    ForStmt {
        var_name: String,
        iterable: Expr,
        body: Vec<Statement>,
    },
    FunctionDecl {
        name: String,
        params: Vec<(String, TypeKind)>,
        return_type: Option<TypeKind>,
        body: Vec<Statement>,
    },
    VizCall {
        viz_type: String, // "plot", "figure", "heatmap", "show"
        args: Vec<Expr>,
    },
    ExprStmt(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub statements: Vec<Statement>,
}

// ============================================================================
// 2. Parser Precedence Levels
// ============================================================================

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
enum Precedence {
    Lowest = 0,
    Assignment = 1,
    LogicalOr = 2,
    LogicalAnd = 3,
    Equality = 4,
    Relational = 5,
    Range = 6,
    Additive = 7,
    Multiplicative = 8,
    MatrixMul = 9,
    Power = 10,
    Unary = 11,
    Postfix = 12,
    Call = 13,
}

impl Precedence {
    fn from_token(kind: &TokenKind) -> Precedence {
        match kind {
            TokenKind::Equal | TokenKind::PlusEqual | TokenKind::MinusEqual | TokenKind::StarEqual | TokenKind::SlashEqual | TokenKind::AtEqual => Precedence::Assignment,
            TokenKind::PipePipe => Precedence::LogicalOr,
            TokenKind::AmpAmp => Precedence::LogicalAnd,
            TokenKind::EqualEqual | TokenKind::NotEqual => Precedence::Equality,
            TokenKind::Less | TokenKind::LessEqual | TokenKind::Greater | TokenKind::GreaterEqual => Precedence::Relational,
            TokenKind::DotDot => Precedence::Range,
            TokenKind::Plus | TokenKind::Minus => Precedence::Additive,
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => Precedence::Multiplicative,
            TokenKind::At => Precedence::MatrixMul,
            TokenKind::Caret => Precedence::Power,
            TokenKind::SingleQuote | TokenKind::Dot => Precedence::Postfix,
            TokenKind::LParen => Precedence::Call,
            _ => Precedence::Lowest,
        }
    }
}

// ============================================================================
// 3. Parser Implementation
// ============================================================================

pub struct Parser<'a> {
    tokens: &'a [Token],
    current: usize,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, current: 0 }
    }

    fn peek(&self) -> &TokenKind {
        self.tokens.get(self.current).map(|t| &t.kind).unwrap_or(&TokenKind::Eof)
    }

    fn peek_token(&self) -> &Token {
        &self.tokens[self.current.min(self.tokens.len() - 1)]
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        &self.tokens[self.current - 1]
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek(), TokenKind::Eof)
    }

    fn expect(&mut self, expected: &TokenKind) -> Result<&Token, String> {
        if self.peek() == expected {
            Ok(self.advance())
        } else {
            let pos = &self.peek_token().position;
            Err(format!(
                "Parse error at line {}, col {}: Expected '{:?}', found '{:?}'",
                pos.line, pos.column, expected, self.peek()
            ))
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, String> {
        let mut statements = Vec::new();
        while !self.is_at_end() {
            statements.push(self.parse_statement()?);
        }
        Ok(Program { statements })
    }

    // ------------------------------------------------------------------------
    // Statement Parsing
    // ------------------------------------------------------------------------
    fn parse_statement(&mut self) -> Result<Statement, String> {
        match self.peek() {
            TokenKind::KeywordLet => self.parse_var_decl(),
            TokenKind::KeywordFn => self.parse_function_decl(),
            TokenKind::KeywordIf => self.parse_if_stmt(),
            TokenKind::KeywordFor => self.parse_for_stmt(),
            TokenKind::KeywordLikh => self.parse_print_stmt(),
            TokenKind::KeywordPlot | TokenKind::KeywordFigure | TokenKind::KeywordHeatmap | TokenKind::KeywordShow => self.parse_viz_call(),
            TokenKind::Ident(_) => {
                // Check if assignment or backward or function call
                if self.peek_next_is_assignment() {
                    self.parse_assignment()
                } else {
                    let expr = self.parse_expression(Precedence::Lowest)?;
                    if self.match_token(&TokenKind::Semicolon) {
                        // Semicolon consumed
                    }
                    if let Expr::MemberAccess { ref object, ref member } = expr {
                        if member == "backward" {
                            return Ok(Statement::BackwardStmt { target: *object.clone() });
                        }
                    }
                    Ok(Statement::ExprStmt(expr))
                }
            }
            _ => {
                let expr = self.parse_expression(Precedence::Lowest)?;
                self.match_token(&TokenKind::Semicolon);
                Ok(Statement::ExprStmt(expr))
            }
        }
    }

    fn peek_next_is_assignment(&self) -> bool {
        let mut offset = 1;
        // Skip identifier
        if let Some(t) = self.tokens.get(self.current + offset) {
            match t.kind {
                TokenKind::Equal | TokenKind::PlusEqual | TokenKind::MinusEqual | TokenKind::StarEqual | TokenKind::SlashEqual | TokenKind::AtEqual => true,
                _ => false,
            }
        } else {
            false
        }
    }

    fn match_token(&mut self, kind: &TokenKind) -> bool {
        if self.peek() == kind {
            self.advance();
            true
        } else {
            false
        }
    }

    fn parse_var_decl(&mut self) -> Result<Statement, String> {
        self.expect(&TokenKind::KeywordLet)?;
        
        let is_mutable = self.match_token(&TokenKind::KeywordMut);

        let name = match self.advance().kind.clone() {
            TokenKind::Ident(s) => s,
            other => return Err(format!("Expected variable name after 'let', found {:?}", other)),
        };

        let type_annotation = if self.match_token(&TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(&TokenKind::Equal)?;
        let initializer = self.parse_expression(Precedence::Lowest)?;
        self.match_token(&TokenKind::Semicolon);

        Ok(Statement::VarDecl {
            name,
            is_mutable,
            type_annotation,
            initializer,
        })
    }

    fn parse_type(&mut self) -> Result<TypeKind, String> {
        match self.advance().kind.clone() {
            TokenKind::KeywordMat => {
                if self.match_token(&TokenKind::LBracket) {
                    let rows = if let TokenKind::Number(n) = self.advance().kind { n as usize } else { 0 };
                    self.expect(&TokenKind::Comma)?;
                    let cols = if let TokenKind::Number(n) = self.advance().kind { n as usize } else { 0 };
                    self.expect(&TokenKind::RBracket)?;
                    Ok(TypeKind::Matrix { rows: Some(rows), cols: Some(cols) })
                } else {
                    Ok(TypeKind::Matrix { rows: None, cols: None })
                }
            }
            TokenKind::KeywordTensor => {
                if self.match_token(&TokenKind::LBracket) {
                    let mut dims = Vec::new();
                    while self.peek() != &TokenKind::RBracket {
                        if let TokenKind::Number(n) = self.advance().kind {
                            dims.push(n as usize);
                        }
                        if self.peek() == &TokenKind::Comma {
                            self.advance();
                        }
                    }
                    self.expect(&TokenKind::RBracket)?;
                    Ok(TypeKind::Tensor { dimensions: dims })
                } else {
                    Ok(TypeKind::Tensor { dimensions: vec![] })
                }
            }
            TokenKind::Ident(name) => match name.as_str() {
                "f64" | "f32" | "float" => Ok(TypeKind::F64),
                "bool" => Ok(TypeKind::Bool),
                "String" | "str" => Ok(TypeKind::String),
                _ => Ok(TypeKind::Custom(name)),
            },
            other => Err(format!("Invalid type annotation {:?}", other)),
        }
    }

    fn parse_assignment(&mut self) -> Result<Statement, String> {
        let target = match self.advance().kind.clone() {
            TokenKind::Ident(s) => s,
            other => return Err(format!("Expected assignment target, found {:?}", other)),
        };

        let op = match self.advance().kind {
            TokenKind::Equal => None,
            TokenKind::PlusEqual => Some(BinaryOpKind::Add),
            TokenKind::MinusEqual => Some(BinaryOpKind::Sub),
            TokenKind::StarEqual => Some(BinaryOpKind::Mul),
            TokenKind::SlashEqual => Some(BinaryOpKind::Div),
            TokenKind::AtEqual => Some(BinaryOpKind::MatMul),
            ref other => return Err(format!("Expected assignment operator, found {:?}", other)),
        };

        let value = self.parse_expression(Precedence::Lowest)?;
        self.match_token(&TokenKind::Semicolon);

        Ok(Statement::Assignment { target, op, value })
    }

    fn parse_print_stmt(&mut self) -> Result<Statement, String> {
        self.expect(&TokenKind::KeywordLikh)?;
        self.expect(&TokenKind::LParen)?;

        let mut args = Vec::new();
        if self.peek() != &TokenKind::RParen {
            loop {
                args.push(self.parse_expression(Precedence::Lowest)?);
                if self.peek() == &TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RParen)?;
        self.match_token(&TokenKind::Semicolon);

        Ok(Statement::PrintStmt { args })
    }

    fn parse_viz_call(&mut self) -> Result<Statement, String> {
        let viz_type = match self.advance().kind.clone() {
            TokenKind::KeywordPlot => "plot".to_string(),
            TokenKind::KeywordFigure => "figure".to_string(),
            TokenKind::KeywordHeatmap => "heatmap".to_string(),
            TokenKind::KeywordShow => "show".to_string(),
            TokenKind::Ident(s) => s,
            other => return Err(format!("Invalid visualization command {:?}", other)),
        };

        let mut args = Vec::new();
        if self.peek() == &TokenKind::LParen {
            self.advance();
            if self.peek() != &TokenKind::RParen {
                loop {
                    args.push(self.parse_expression(Precedence::Lowest)?);
                    if self.peek() == &TokenKind::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect(&TokenKind::RParen)?;
        }
        self.match_token(&TokenKind::Semicolon);

        Ok(Statement::VizCall { viz_type, args })
    }

    fn parse_if_stmt(&mut self) -> Result<Statement, String> {
        self.expect(&TokenKind::KeywordIf)?;
        let condition = self.parse_expression(Precedence::Lowest)?;

        self.expect(&TokenKind::LBrace)?;
        let mut then_branch = Vec::new();
        while self.peek() != &TokenKind::RBrace && !self.is_at_end() {
            then_branch.push(self.parse_statement()?);
        }
        self.expect(&TokenKind::RBrace)?;

        let else_branch = if self.match_token(&TokenKind::KeywordElse) {
            self.expect(&TokenKind::LBrace)?;
            let mut el = Vec::new();
            while self.peek() != &TokenKind::RBrace && !self.is_at_end() {
                el.push(self.parse_statement()?);
            }
            self.expect(&TokenKind::RBrace)?;
            Some(el)
        } else {
            None
        };

        Ok(Statement::IfStmt {
            condition,
            then_branch,
            else_branch,
        })
    }

    fn parse_for_stmt(&mut self) -> Result<Statement, String> {
        self.expect(&TokenKind::KeywordFor)?;
        let var_name = match self.advance().kind.clone() {
            TokenKind::Ident(s) => s,
            other => return Err(format!("Expected variable name in for-loop, found {:?}", other)),
        };
        self.expect(&TokenKind::KeywordIn)?;
        let iterable = self.parse_expression(Precedence::Lowest)?;

        self.expect(&TokenKind::LBrace)?;
        let mut body = Vec::new();
        while self.peek() != &TokenKind::RBrace && !self.is_at_end() {
            body.push(self.parse_statement()?);
        }
        self.expect(&TokenKind::RBrace)?;

        Ok(Statement::ForStmt {
            var_name,
            iterable,
            body,
        })
    }

    fn parse_function_decl(&mut self) -> Result<Statement, String> {
        self.expect(&TokenKind::KeywordFn)?;
        let name = match self.advance().kind.clone() {
            TokenKind::Ident(s) => s,
            other => return Err(format!("Expected function name after 'fn', found {:?}", other)),
        };

        self.expect(&TokenKind::LParen)?;
        let mut params = Vec::new();
        if self.peek() != &TokenKind::RParen {
            loop {
                let p_name = match self.advance().kind.clone() {
                    TokenKind::Ident(s) => s,
                    other => return Err(format!("Expected parameter name, found {:?}", other)),
                };
                self.expect(&TokenKind::Colon)?;
                let p_type = self.parse_type()?;
                params.push((p_name, p_type));

                if self.peek() == &TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RParen)?;

        let return_type = if self.match_token(&TokenKind::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(&TokenKind::LBrace)?;
        let mut body = Vec::new();
        while self.peek() != &TokenKind::RBrace && !self.is_at_end() {
            body.push(self.parse_statement()?);
        }
        self.expect(&TokenKind::RBrace)?;

        Ok(Statement::FunctionDecl {
            name,
            params,
            return_type,
            body,
        })
    }

    // ------------------------------------------------------------------------
    // Expression Parsing (Pratt Parser)
    // ------------------------------------------------------------------------
    fn parse_expression(&mut self, precedence: Precedence) -> Result<Expr, String> {
        let mut left = self.parse_prefix()?;

        while !self.is_at_end() && precedence < Precedence::from_token(self.peek()) {
            left = self.parse_infix(left)?;
        }

        Ok(left)
    }

    fn parse_prefix(&mut self) -> Result<Expr, String> {
        let token = self.advance().clone();
        match token.kind {
            TokenKind::Number(n) => Ok(Expr::Number(n)),
            TokenKind::StringLiteral(s) => Ok(Expr::StringLiteral(s)),
            TokenKind::BoolLiteral(b) => Ok(Expr::BoolLiteral(b)),
            TokenKind::Ident(s) => Ok(Expr::Identifier(s)),
            TokenKind::LParen => {
                let expr = self.parse_expression(Precedence::Lowest)?;
                self.expect(&TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::LBracket => self.parse_matrix_or_tensor_literal(),
            TokenKind::Minus => {
                let expr = self.parse_expression(Precedence::Unary)?;
                Ok(Expr::UnaryOp {
                    op: UnaryOpKind::Neg,
                    expr: Box::new(expr),
                })
            }
            TokenKind::Bang => {
                let expr = self.parse_expression(Precedence::Unary)?;
                Ok(Expr::UnaryOp {
                    op: UnaryOpKind::Not,
                    expr: Box::new(expr),
                })
            }
            ref other => Err(format!(
                "Parse error at line {}, col {}: Unexpected prefix token {:?}",
                token.position.line, token.position.column, other
            )),
        }
    }

    fn parse_matrix_or_tensor_literal(&mut self) -> Result<Expr, String> {
        // Parse nested brackets [[1.0, 2.0], [3.0, 4.0]]
        let mut rows = Vec::new();
        if self.peek() == &TokenKind::RBracket {
            self.advance();
            return Ok(Expr::MatrixLiteral(vec![]));
        }

        while self.peek() != &TokenKind::RBracket {
            if self.peek() == &TokenKind::LBracket {
                self.advance(); // consume inner '['
                let mut row = Vec::new();
                while self.peek() != &TokenKind::RBracket {
                    row.push(self.parse_expression(Precedence::Lowest)?);
                    if self.peek() == &TokenKind::Comma {
                        self.advance();
                    }
                }
                self.expect(&TokenKind::RBracket)?;
                rows.push(row);
            } else {
                // 1D Array literal
                let mut single_row = Vec::new();
                single_row.push(self.parse_expression(Precedence::Lowest)?);
                while self.peek() == &TokenKind::Comma {
                    self.advance();
                    single_row.push(self.parse_expression(Precedence::Lowest)?);
                }
                rows.push(single_row);
                break;
            }

            if self.peek() == &TokenKind::Comma {
                self.advance();
            }
        }
        self.expect(&TokenKind::RBracket)?;

        Ok(Expr::MatrixLiteral(rows))
    }

    fn parse_infix(&mut self, left: Expr) -> Result<Expr, String> {
        let token = self.advance().clone();
        let precedence = Precedence::from_token(&token.kind);

        match token.kind {
            TokenKind::Plus => self.binary_op(left, BinaryOpKind::Add, precedence),
            TokenKind::Minus => self.binary_op(left, BinaryOpKind::Sub, precedence),
            TokenKind::Star => self.binary_op(left, BinaryOpKind::Mul, precedence),
            TokenKind::Slash => self.binary_op(left, BinaryOpKind::Div, precedence),
            TokenKind::Percent => self.binary_op(left, BinaryOpKind::Mod, precedence),
            TokenKind::Caret => self.binary_op(left, BinaryOpKind::Pow, Precedence::Power), // Right associative
            TokenKind::At => self.binary_op(left, BinaryOpKind::MatMul, precedence),
            TokenKind::EqualEqual => self.binary_op(left, BinaryOpKind::Eq, precedence),
            TokenKind::NotEqual => self.binary_op(left, BinaryOpKind::Neq, precedence),
            TokenKind::Less => self.binary_op(left, BinaryOpKind::Lt, precedence),
            TokenKind::LessEqual => self.binary_op(left, BinaryOpKind::Lte, precedence),
            TokenKind::Greater => self.binary_op(left, BinaryOpKind::Gt, precedence),
            TokenKind::GreaterEqual => self.binary_op(left, BinaryOpKind::Gte, precedence),
            TokenKind::AmpAmp => self.binary_op(left, BinaryOpKind::And, precedence),
            TokenKind::PipePipe => self.binary_op(left, BinaryOpKind::Or, precedence),
            TokenKind::DotDot => {
                let right = self.parse_expression(precedence)?;
                Ok(Expr::Range {
                    start: Box::new(left),
                    end: Box::new(right),
                })
            }
            TokenKind::SingleQuote => {
                // Postfix Transpose Operator A'
                Ok(Expr::UnaryOp {
                    op: UnaryOpKind::Transpose,
                    expr: Box::new(left),
                })
            }
            TokenKind::Dot => {
                // Member Access or Method Call
                let member = match self.advance().kind.clone() {
                    TokenKind::Ident(s) => s,
                    TokenKind::KeywordBackward => "backward".to_string(),
                    other => return Err(format!("Expected member identifier after '.', found {:?}", other)),
                };

                if self.peek() == &TokenKind::LParen {
                    self.advance(); // consume '('
                    let mut args = Vec::new();
                    if self.peek() != &TokenKind::RParen {
                        loop {
                            args.push(self.parse_expression(Precedence::Lowest)?);
                            if self.peek() == &TokenKind::Comma {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(&TokenKind::RParen)?;
                    let callee_name = match left {
                        Expr::Identifier(id) => format!("{}.{}", id, member),
                        _ => member,
                    };
                    Ok(Expr::FunctionCall {
                        callee: callee_name,
                        args,
                    })
                } else {
                    Ok(Expr::MemberAccess {
                        object: Box::new(left),
                        member,
                    })
                }
            }
            TokenKind::DoubleColon => {
                // Module function call e.g. attn::self_attention
                let member = match self.advance().kind.clone() {
                    TokenKind::Ident(s) => s,
                    other => return Err(format!("Expected function name after '::', found {:?}", other)),
                };
                let callee_name = match left {
                    Expr::Identifier(id) => format!("{}::{}", id, member),
                    _ => member,
                };
                self.expect(&TokenKind::LParen)?;
                let mut args = Vec::new();
                if self.peek() != &TokenKind::RParen {
                    loop {
                        args.push(self.parse_expression(Precedence::Lowest)?);
                        if self.peek() == &TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&TokenKind::RParen)?;

                Ok(Expr::FunctionCall {
                    callee: callee_name,
                    args,
                })
            }
            TokenKind::LParen => {
                // Function call on identifier e.g. foo(x, y)
                let mut args = Vec::new();
                if self.peek() != &TokenKind::RParen {
                    loop {
                        args.push(self.parse_expression(Precedence::Lowest)?);
                        if self.peek() == &TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&TokenKind::RParen)?;

                let callee_name = match left {
                    Expr::Identifier(id) => id,
                    _ => return Err("Invalid callee for function call".into()),
                };

                Ok(Expr::FunctionCall {
                    callee: callee_name,
                    args,
                })
            }
            ref other => Err(format!("Unexpected infix operator {:?}", other)),
        }
    }

    fn binary_op(&mut self, left: Expr, op: BinaryOpKind, precedence: Precedence) -> Result<Expr, String> {
        let right = self.parse_expression(precedence)?;
        Ok(Expr::BinaryOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        })
    }
}

/// Main entry point to parse tokens into a Phaadu AST Program
pub fn parse(tokens: &[Token]) -> Result<Program, String> {
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

// ============================================================================
// 4. Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;

    #[test]
    fn test_parse_matrix_multiplication_and_transpose() {
        let src = "let C = A @ B';";
        let tokens = lex(src).unwrap();
        let program = parse(&tokens).unwrap();

        assert_eq!(program.statements.len(), 1);
        if let Statement::VarDecl { name, initializer, .. } = &program.statements[0] {
            assert_eq!(name, "C");
            if let Expr::BinaryOp { left, op, right } = initializer {
                assert_eq!(**left, Expr::Identifier("A".to_string()));
                assert_eq!(*op, BinaryOpKind::MatMul);
                assert_eq!(**right, Expr::UnaryOp {
                    op: UnaryOpKind::Transpose,
                    expr: Box::new(Expr::Identifier("B".to_string()))
                });
            } else {
                panic!("Expected binary matrix multiplication operation");
            }
        }
    }

    #[test]
    fn test_parse_attention_and_autodiff() {
        let src = r#"
            let out = attn::self_attention(Q, K, V);
            loss.backward();
            likh("Done", out);
        "#;
        let tokens = lex(src).unwrap();
        let program = parse(&tokens).unwrap();

        assert_eq!(program.statements.len(), 3);

        // Check statement 1: self-attention function call
        if let Statement::VarDecl { name, initializer, .. } = &program.statements[0] {
            assert_eq!(name, "out");
            if let Expr::FunctionCall { callee, args } = initializer {
                assert_eq!(callee, "attn::self_attention");
                assert_eq!(args.len(), 3);
            }
        }

        // Check statement 2: backward statement
        if let Statement::BackwardStmt { target } = &program.statements[1] {
            assert_eq!(*target, Expr::Identifier("loss".to_string()));
        }

        // Check statement 3: print call
        if let Statement::PrintStmt { args } = &program.statements[2] {
            assert_eq!(args.len(), 2);
        }
    }
}

