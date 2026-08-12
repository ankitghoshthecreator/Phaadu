mod lexer;
mod parser;
mod runtime;

fn main() {
    let phaadu_source = r#"
        // Phaadu Language Code Example
        let A: Matrix[2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let B: Matrix[3, 2] = [[7.0, 8.0], [9.0, 1.0], [2.0, 3.0]];
        
        let C = A @ B'; // Matrix multiplication with transpose
        let attn_score = attn::self_attention(Q, K, V);
        
        loss.backward();

        likh("Computed Matrix C:", C);
        viz::plot(epochs, loss_history);
    "#;

    println!("--- ⚡ Phaadu Compiler Pipeline ---");

    // 1. Lexical Analysis (Part 1: phaadu-lexer)
    let tokens = match lexer::lex(phaadu_source) {
        Ok(t) => {
            println!("✅ [Part 1: Lexer] Tokenized {} tokens successfully!", t.len());
            t
        }
        Err(e) => {
            eprintln!("❌ Lexer error: {}", e);
            return;
        }
    };

    // 2. Syntax Analysis & AST Generation (Part 2: phaadu-parser)
    match parser::parse(&tokens) {
        Ok(program) => {
            println!("✅ [Part 2: Parser] Parsed {} AST statements successfully!\n", program.statements.len());
            println!("--- 🌳 Abstract Syntax Tree (AST) Summary ---");
            for (idx, stmt) in program.statements.iter().enumerate() {
                println!("Statement {}: {:#?}", idx + 1, stmt);
            }
        }
        Err(e) => {
            eprintln!("❌ Parse error: {}", e);
        }
    }
}


