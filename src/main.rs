mod lexer;
mod parser;
mod analyzer;
mod autodiff;
mod runtime;

fn main() {
    let phaadu_source = r#"
        // Phaadu Language Code Example
        let A: Matrix[2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let B: Matrix[3, 2] = [[7.0, 8.0], [9.0, 1.0], [2.0, 3.0]];
        
        let C = A @ B; // Matrix multiplication
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
    let program = match parser::parse(&tokens) {
        Ok(program) => {
            println!("✅ [Part 2: Parser] Parsed {} AST statements successfully!\n", program.statements.len());
            program
        }
        Err(e) => {
            eprintln!("❌ Parse error: {}", e);
            return;
        }
    };

    // 3. Semantic Analysis & Shape Inference (Part 3: phaadu-analyzer)
    match analyzer::analyze(&program) {
        Ok(_) => {
            println!("✅ [Part 3: Analyzer] Semantic analysis and shape inference successful!");
        }
        Err(e) => {
            eprintln!("❌ Semantic analysis error: {}", e);
            return;
        }
    }

    // 4. Automatic Differentiation (Part 4: phaadu-autodiff)
    match autodiff::generate_backward(&program) {
        Ok(backward_program) => {
            println!("✅ [Part 4: Autodiff] Static reverse-mode automatic differentiation successful!");
            println!("Generated backward pass statements: {} total statements.", backward_program.statements.len());
        }
        Err(e) => {
            eprintln!("❌ Autodiff error: {}", e);
        }
    }
}


