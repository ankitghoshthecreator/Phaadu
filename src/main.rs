// Step 1: pretend we "read" source code
fn main() {
    let source = "A = B * C + 5";

    // Step 2: our fake lexer just splits on spaces
    let tokens: Vec<&str> = source.split_whitespace().collect();
    println!("Tokens: {:?}", tokens);

    // Step 3: fake "parser" prints a structure
    println!("Parsed as: (Assign A (Add (Mul B C) 5))");

    // Step 4: pretend we "compiled" to assembly
    println!("Assembly:");
    println!("  mov r0, [B]");
    println!("  mul r0, [C]");
    println!("  add r0, 5");
    println!("  mov [A], r0");
}
