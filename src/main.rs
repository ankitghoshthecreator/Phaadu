mod lexer;
mod parser;
mod runtime;

fn main() {
    let phaadu_source = r#"
        // Phaadu Language Example: Matrix Multiplication & Self-Attention
        let A: Matrix[2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let B: Matrix[3, 2] = [[7.0, 8.0], [9.0, 1.0], [2.0, 3.0]];
        
        let C = A @ B; // Matrix multiplication using @
        let C_transposed = C';

        // Neural Net Attention & Autodiff
        let W = tensor::randn([64, 32], requires_grad=true);
        let attn_score = attn::self_attention(Q, K, V);
        loss.backward();

        // Print & Plotting Output
        likh("Matrix C:");
        likh(C);
        viz::plot(epochs, loss_history);
    "#;

    println!("--- ⚡ Phaadu Compiler (Part 1: Lexer & Tokenizer) ---");

    // 1. Tokenize using phaadu-lexer
    match lexer::lex(phaadu_source) {
        Ok(tokens) => {
            println!("Successfully tokenized {} tokens!", tokens.len());
            println!("\nSample Token Output:");
            for token in tokens.iter().take(15) {
                println!("  Line {:2}, Col {:2} | {:?}", token.position.line, token.position.column, token.kind);
            }
        }
        Err(e) => {
            eprintln!("Lexer error: {}", e);
        }
    }
}

