mod lexer;
mod parser;
mod runtime;

fn main() {
    let src = r#"likh("Hello, parser!")"#;

    // 1. tokenize
    let tokens = match lexer::lex(src) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Lexing error: {}", e);
            return;
        }
    };

    println!("Tokens: {:?}", tokens);

    // 2. parse
    match parser::parse(&tokens) {
        Ok(ast) => println!("AST: {:?}", ast),
        Err(e) => eprintln!("Parse error: {}", e),
    }
}
