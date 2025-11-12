use crate::parser::AstNode;

/// Interpret and execute an AST node.
pub fn run(ast: &AstNode) -> Result<(), String> {
    match ast {
        AstNode::PrintCall(text) => {
            println!("{}", text);
            Ok(())
        }
    }
}
