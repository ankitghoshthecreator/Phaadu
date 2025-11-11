mod lexer;

fn main() {
    let src = r#"likh("Hello, world!")"#;

    match lexer::lex(src) {
        Ok(tokens) => {
            println!("Tokens:");
            for t in tokens {
                println!("{:?}", t);
            }
        }
        Err(e) => eprintln!("Lexing error: {}", e),
    }
}
