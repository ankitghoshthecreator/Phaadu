use crate::parser::{Program, Statement, Expr, BinaryOpKind, UnaryOpKind, TypeKind};
use std::collections::{HashMap, HashSet};

/// Helper to extract shape dimensions array `[d1, d2, ...]` from a nested matrix literal.
fn extract_dims_from_expr(expr: &Expr) -> Option<Vec<usize>> {
    match expr {
        Expr::MatrixLiteral(rows) => {
            if rows.is_empty() {
                return Some(vec![]);
            }
            if rows.len() == 1 {
                let mut dims = Vec::new();
                for elem in &rows[0] {
                    if let Expr::Number(n) = elem {
                        dims.push(*n as usize);
                    } else {
                        return None;
                    }
                }
                Some(dims)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Recursively infers the shape of an expression based on variable shapes.
fn infer_shape(expr: &Expr, var_shapes: &HashMap<String, Vec<usize>>) -> Option<Vec<usize>> {
    match expr {
        Expr::Number(_) => Some(vec![]),
        Expr::Identifier(name) => var_shapes.get(name).cloned(),
        Expr::BinaryOp { left, op, right } => {
            let left_shape = infer_shape(left, var_shapes);
            let right_shape = infer_shape(right, var_shapes);
            match op {
                BinaryOpKind::Add | BinaryOpKind::Sub | BinaryOpKind::Mul | BinaryOpKind::Div | BinaryOpKind::Pow => {
                    match (left_shape, right_shape) {
                        (Some(l), Some(r)) => {
                            if l.is_empty() { Some(r) } else { Some(l) }
                        }
                        (Some(l), None) => Some(l),
                        (None, Some(r)) => Some(r),
                        _ => None,
                    }
                }
                BinaryOpKind::MatMul => {
                    match (left_shape, right_shape) {
                        (Some(l), Some(r)) => {
                            if l.len() >= 2 && r.len() >= 2 {
                                let mut res = l[..l.len() - 2].to_vec();
                                res.push(l[l.len() - 2]);
                                res.push(r[r.len() - 1]);
                                Some(res)
                            } else if l.len() == 2 && r.len() == 2 {
                                Some(vec![l[0], r[1]])
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
        }
        Expr::UnaryOp { op, expr } => {
            let shape = infer_shape(expr, var_shapes)?;
            match op {
                UnaryOpKind::Transpose => {
                    if shape.len() >= 2 {
                        let mut res = shape.clone();
                        let l = res.len();
                        res.swap(l - 1, l - 2);
                        Some(res)
                    } else {
                        Some(shape)
                    }
                }
                _ => Some(shape),
            }
        }
        Expr::FunctionCall { callee, args } => {
            if callee == "randn" || callee == "tensor::randn" || callee == "tensor::ones" || callee == "tensor::zeros" {
                if !args.is_empty() {
                    extract_dims_from_expr(&args[0])
                } else {
                    Some(vec![])
                }
            } else if callee == "attn::self_attention" || callee == "attn::cross_attention" {
                if !args.is_empty() {
                    infer_shape(&args[0], var_shapes)
                } else {
                    None
                }
            } else {
                None
            }
        }
        Expr::NamedArg { value, .. } => {
            infer_shape(value, var_shapes)
        }
        _ => None,
    }
}

/// Infers the total element size of an expression shape.
fn infer_shape_size(expr: &Expr, var_shapes: &HashMap<String, Vec<usize>>) -> usize {
    if let Some(shape) = infer_shape(expr, var_shapes) {
        let size = shape.iter().product::<usize>();
        if size == 0 { 1 } else { size }
    } else {
        1
    }
}

/// Recursively checks if an expression contains `requires_grad=true`.
fn has_requires_grad(expr: &Expr) -> bool {
    match expr {
        Expr::NamedArg { name, value } => {
            if name == "requires_grad" {
                if let Expr::BoolLiteral(b) = **value {
                    return b;
                }
            }
            has_requires_grad(value)
        }
        Expr::BinaryOp { left, right, .. } => {
            has_requires_grad(left) || has_requires_grad(right)
        }
        Expr::UnaryOp { expr, .. } => {
            has_requires_grad(expr)
        }
        Expr::FunctionCall { args, .. } => {
            args.iter().any(has_requires_grad)
        }
        Expr::MatrixLiteral(rows) => {
            rows.iter().any(|row| row.iter().any(has_requires_grad))
        }
        Expr::Range { start, end } => {
            has_requires_grad(start) || has_requires_grad(end)
        }
        _ => false,
    }
}

/// Checks if an expression references any variable that requires gradients.
fn depends_on_vars(expr: &Expr, requires_grad_vars: &HashSet<String>) -> bool {
    match expr {
        Expr::Identifier(name) => requires_grad_vars.contains(name),
        Expr::BinaryOp { left, right, .. } => {
            depends_on_vars(left, requires_grad_vars) || depends_on_vars(right, requires_grad_vars)
        }
        Expr::UnaryOp { expr, .. } => {
            depends_on_vars(expr, requires_grad_vars)
        }
        Expr::FunctionCall { args, .. } => {
            args.iter().any(|arg| depends_on_vars(arg, requires_grad_vars))
        }
        Expr::MatrixLiteral(rows) => {
            rows.iter().any(|row| row.iter().any(|elem| depends_on_vars(elem, requires_grad_vars)))
        }
        Expr::Range { start, end } => {
            depends_on_vars(start, requires_grad_vars) || depends_on_vars(end, requires_grad_vars)
        }
        Expr::NamedArg { value, .. } => {
            depends_on_vars(value, requires_grad_vars)
        }
        Expr::MemberAccess { object, .. } => {
            depends_on_vars(object, requires_grad_vars)
        }
        _ => false,
    }
}

/// Rewrites member access expressions like `v.grad` to `v_grad`.
fn rewrite_grad_accesses(expr: &mut Expr) {
    match expr {
        Expr::MemberAccess { object, member } => {
            if member == "grad" {
                if let Expr::Identifier(name) = &**object {
                    *expr = Expr::Identifier(format!("{}_grad", name));
                    return;
                }
            }
            rewrite_grad_accesses(object);
        }
        Expr::BinaryOp { left, right, .. } => {
            rewrite_grad_accesses(left);
            rewrite_grad_accesses(right);
        }
        Expr::UnaryOp { expr, .. } => {
            rewrite_grad_accesses(expr);
        }
        Expr::FunctionCall { args, .. } => {
            for arg in args {
                rewrite_grad_accesses(arg);
            }
        }
        Expr::MatrixLiteral(rows) => {
            for row in rows {
                for elem in row {
                    rewrite_grad_accesses(elem);
                }
            }
        }
        Expr::Range { start, end } => {
            rewrite_grad_accesses(start);
            rewrite_grad_accesses(end);
        }
        Expr::NamedArg { value, .. } => {
            rewrite_grad_accesses(value);
        }
        _ => {}
    }
}

/// Recursively walks statements to rewrite any `.grad` member accesses.
fn rewrite_statement_grads(stmt: &mut Statement) {
    match stmt {
        Statement::VarDecl { initializer, .. } => {
            rewrite_grad_accesses(initializer);
        }
        Statement::Assignment { value, .. } => {
            rewrite_grad_accesses(value);
        }
        Statement::PrintStmt { args } => {
            for arg in args {
                rewrite_grad_accesses(arg);
            }
        }
        Statement::BackwardStmt { target } => {
            rewrite_grad_accesses(target);
        }
        Statement::IfStmt { condition, then_branch, else_branch } => {
            rewrite_grad_accesses(condition);
            for s in then_branch {
                rewrite_statement_grads(s);
            }
            if let Some(el) = else_branch {
                for s in el {
                    rewrite_statement_grads(s);
                }
            }
        }
        Statement::ForStmt { iterable, body, .. } => {
            rewrite_grad_accesses(iterable);
            for s in body {
                rewrite_statement_grads(s);
            }
        }
        Statement::FunctionDecl { body, .. } => {
            for s in body {
                rewrite_statement_grads(s);
            }
        }
        Statement::VizCall { args, .. } => {
            for arg in args {
                rewrite_grad_accesses(arg);
            }
        }
        Statement::ExprStmt(expr) => {
            rewrite_grad_accesses(expr);
        }
    }
}

/// Propagates the upstream gradient `dy` to the variables inside an expression.
fn propagate(
    expr: &Expr,
    dy: Expr,
    accumulated_grads: &mut HashMap<String, Vec<Expr>>,
    requires_grad_vars: &HashSet<String>,
    var_shapes: &HashMap<String, Vec<usize>>,
) {
    match expr {
        Expr::Identifier(name) => {
            if requires_grad_vars.contains(name) {
                accumulated_grads.entry(name.clone()).or_insert_with(Vec::new).push(dy);
            }
        }
        Expr::BinaryOp { left, op, right } => {
            match op {
                BinaryOpKind::Add => {
                    propagate(left, dy.clone(), accumulated_grads, requires_grad_vars, var_shapes);
                    propagate(right, dy, accumulated_grads, requires_grad_vars, var_shapes);
                }
                BinaryOpKind::Sub => {
                    propagate(left, dy.clone(), accumulated_grads, requires_grad_vars, var_shapes);
                    let neg_dy = Expr::UnaryOp {
                        op: UnaryOpKind::Neg,
                        expr: Box::new(dy),
                    };
                    propagate(right, neg_dy, accumulated_grads, requires_grad_vars, var_shapes);
                }
                BinaryOpKind::Mul => {
                    let left_dy = Expr::BinaryOp {
                        left: Box::new(dy.clone()),
                        op: BinaryOpKind::Mul,
                        right: right.clone(),
                    };
                    let right_dy = Expr::BinaryOp {
                        left: Box::new(dy),
                        op: BinaryOpKind::Mul,
                        right: left.clone(),
                    };
                    propagate(left, left_dy, accumulated_grads, requires_grad_vars, var_shapes);
                    propagate(right, right_dy, accumulated_grads, requires_grad_vars, var_shapes);
                }
                BinaryOpKind::Div => {
                    let left_dy = Expr::BinaryOp {
                        left: Box::new(dy.clone()),
                        op: BinaryOpKind::Div,
                        right: right.clone(),
                    };
                    let right_dy = Expr::BinaryOp {
                        left: Box::new(Expr::UnaryOp {
                            op: UnaryOpKind::Neg,
                            expr: Box::new(Expr::BinaryOp {
                                left: Box::new(dy),
                                op: BinaryOpKind::Mul,
                                right: left.clone(),
                            }),
                        }),
                        op: BinaryOpKind::Div,
                        right: Box::new(Expr::BinaryOp {
                            left: right.clone(),
                            op: BinaryOpKind::Pow,
                            right: Box::new(Expr::Number(2.0)),
                        }),
                    };
                    propagate(left, left_dy, accumulated_grads, requires_grad_vars, var_shapes);
                    propagate(right, right_dy, accumulated_grads, requires_grad_vars, var_shapes);
                }
                BinaryOpKind::Pow => {
                    if let Expr::Number(n) = &**right {
                        let deriv = Expr::BinaryOp {
                            left: Box::new(Expr::Number(*n)),
                            op: BinaryOpKind::Mul,
                            right: Box::new(Expr::BinaryOp {
                                left: left.clone(),
                                op: BinaryOpKind::Pow,
                                right: Box::new(Expr::Number(*n - 1.0)),
                            }),
                        };
                        let left_dy = Expr::BinaryOp {
                            left: Box::new(dy),
                            op: BinaryOpKind::Mul,
                            right: Box::new(deriv),
                        };
                        propagate(left, left_dy, accumulated_grads, requires_grad_vars, var_shapes);
                    }
                }
                BinaryOpKind::MatMul => {
                    let left_dy = Expr::BinaryOp {
                        left: Box::new(dy.clone()),
                        op: BinaryOpKind::MatMul,
                        right: Box::new(Expr::UnaryOp {
                            op: UnaryOpKind::Transpose,
                            expr: right.clone(),
                        }),
                    };
                    let right_dy = Expr::BinaryOp {
                        left: Box::new(Expr::UnaryOp {
                            op: UnaryOpKind::Transpose,
                            expr: left.clone(),
                        }),
                        op: BinaryOpKind::MatMul,
                        right: Box::new(dy),
                    };
                    propagate(left, left_dy, accumulated_grads, requires_grad_vars, var_shapes);
                    propagate(right, right_dy, accumulated_grads, requires_grad_vars, var_shapes);
                }
                _ => {}
            }
        }
        Expr::UnaryOp { op, expr: sub_expr } => {
            match op {
                UnaryOpKind::Neg => {
                    let neg_dy = Expr::UnaryOp {
                        op: UnaryOpKind::Neg,
                        expr: Box::new(dy),
                    };
                    propagate(sub_expr, neg_dy, accumulated_grads, requires_grad_vars, var_shapes);
                }
                UnaryOpKind::Transpose => {
                    let trans_dy = Expr::UnaryOp {
                        op: UnaryOpKind::Transpose,
                        expr: Box::new(dy),
                    };
                    propagate(sub_expr, trans_dy, accumulated_grads, requires_grad_vars, var_shapes);
                }
                _ => {}
            }
        }
        Expr::FunctionCall { callee, args } => {
            if callee == "mean" && !args.is_empty() {
                let size = infer_shape_size(&args[0], var_shapes);
                let mean_dy = Expr::BinaryOp {
                    left: Box::new(dy),
                    op: BinaryOpKind::Mul,
                    right: Box::new(Expr::Number(1.0 / size as f64)),
                };
                propagate(&args[0], mean_dy, accumulated_grads, requires_grad_vars, var_shapes);
            }
        }
        Expr::NamedArg { value, .. } => {
            propagate(value, dy, accumulated_grads, requires_grad_vars, var_shapes);
        }
        _ => {}
    }
}

/// Static Reverse-Mode Autodiff Pass.
/// Scans the program, builds the computation graph, emits backward pass statements when encountering
/// `.backward()`, and rewrites `.grad` variable accesses to reference the newly generated gradient variables.
pub fn generate_backward(program: &Program) -> Result<Program, String> {
    let mut new_statements = Vec::new();
    let mut var_shapes = HashMap::new();
    let mut requires_grad_vars = HashSet::new();

    for stmt in &program.statements {
        match stmt {
            Statement::VarDecl { name, is_mutable, type_annotation, initializer } => {
                let shape = if let Some(annot) = type_annotation {
                    match annot {
                        TypeKind::Matrix { rows, cols } => Some(vec![rows.unwrap_or(1), cols.unwrap_or(1)]),
                        TypeKind::Tensor { dimensions } => Some(dimensions.clone()),
                        _ => Some(vec![]),
                    }
                } else {
                    infer_shape(initializer, &var_shapes)
                };
                if let Some(s) = shape {
                    var_shapes.insert(name.clone(), s);
                }

                if has_requires_grad(initializer) {
                    requires_grad_vars.insert(name.clone());
                } else if depends_on_vars(initializer, &requires_grad_vars) {
                    requires_grad_vars.insert(name.clone());
                }

                new_statements.push(stmt.clone());
            }
            Statement::Assignment { target, op: _, value } => {
                if depends_on_vars(value, &requires_grad_vars) {
                    requires_grad_vars.insert(target.clone());
                }
                new_statements.push(stmt.clone());
            }
            Statement::BackwardStmt { target } => {
                let target_name = match target {
                    Expr::Identifier(name) => name.clone(),
                    _ => return Err("Backward pass target must be a variable".to_string()),
                };

                let mut accumulated_grads: HashMap<String, Vec<Expr>> = HashMap::new();
                accumulated_grads.insert(target_name.clone(), vec![Expr::Number(1.0)]);

                let mut backward_statements = Vec::new();

                for f_stmt in new_statements.iter().rev() {
                    match f_stmt {
                        Statement::VarDecl { name, initializer, .. } | Statement::Assignment { target: name, value: initializer, .. } => {
                            if requires_grad_vars.contains(name) {
                                if let Some(grads) = accumulated_grads.remove(name) {
                                    let sum_grad = if grads.len() == 1 {
                                        grads[0].clone()
                                    } else {
                                        let mut iter = grads.into_iter();
                                        let mut acc = iter.next().unwrap();
                                        for g in iter {
                                            acc = Expr::BinaryOp {
                                                left: Box::new(acc),
                                                op: BinaryOpKind::Add,
                                                right: Box::new(g),
                                            };
                                        }
                                        acc
                                    };

                                    let grad_var_name = format!("{}_grad", name);
                                    
                                    backward_statements.push(Statement::VarDecl {
                                        name: grad_var_name.clone(),
                                        is_mutable: false,
                                        type_annotation: None,
                                        initializer: sum_grad.clone(),
                                    });

                                    let upstream = Expr::Identifier(grad_var_name);
                                    propagate(initializer, upstream, &mut accumulated_grads, &requires_grad_vars, &var_shapes);
                                }
                            }
                        }
                        _ => {}
                    }
                }

                for req_var in &requires_grad_vars {
                    let grad_var_name = format!("{}_grad", req_var);
                    if !backward_statements.iter().any(|s| {
                        if let Statement::VarDecl { name, .. } = s {
                            name == &grad_var_name
                        } else {
                            false
                        }
                    }) {
                        let shape = var_shapes.get(req_var).cloned().unwrap_or(vec![]);
                        let zeros_expr = if shape.is_empty() {
                            Expr::Number(0.0)
                        } else {
                            Expr::FunctionCall {
                                callee: "tensor::zeros".to_string(),
                                args: vec![Expr::MatrixLiteral(vec![shape.into_iter().map(|s| Expr::Number(s as f64)).collect()])],
                            }
                        };
                        backward_statements.push(Statement::VarDecl {
                            name: grad_var_name,
                            is_mutable: false,
                            type_annotation: None,
                            initializer: zeros_expr,
                        });
                    }
                }

                for b_stmt in backward_statements {
                    new_statements.push(b_stmt);
                }
            }
            _ => {
                let mut stmt_copy = stmt.clone();
                rewrite_statement_grads(&mut stmt_copy);
                new_statements.push(stmt_copy);
            }
        }
    }

    Ok(Program { statements: new_statements })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    #[test]
    fn test_autodiff_simple_scalar() {
        let src = r#"
            let x = tensor::randn([1], requires_grad=true);
            let y = x * 2.0;
            let loss = y + 5.0;
            loss.backward();
            likh(x.grad);
        "#;
        let tokens = lex(src).unwrap();
        let program = parse(&tokens).unwrap();
        let result = generate_backward(&program).unwrap();

        // Let's verify that the output program contains:
        // - loss_grad declaration (initialized to 1.0)
        // - y_grad declaration (initialized to loss_grad)
        // - x_grad declaration (initialized to y_grad * 2.0)
        // - the print statement rewritten to reference x_grad
        let mut has_loss_grad = false;
        let mut has_y_grad = false;
        let mut has_x_grad = false;
        let mut has_likh_x_grad = false;

        for stmt in &result.statements {
            match stmt {
                Statement::VarDecl { name, initializer, .. } => {
                    if name == "loss_grad" {
                        has_loss_grad = true;
                        assert_eq!(*initializer, Expr::Number(1.0));
                    } else if name == "y_grad" {
                        has_y_grad = true;
                        assert_eq!(*initializer, Expr::Identifier("loss_grad".to_string()));
                    } else if name == "x_grad" {
                        has_x_grad = true;
                        if let Expr::BinaryOp { left, op, right } = initializer {
                            assert_eq!(**left, Expr::Identifier("y_grad".to_string()));
                            assert_eq!(*op, BinaryOpKind::Mul);
                            assert_eq!(**right, Expr::Number(2.0));
                        } else {
                            panic!("x_grad initializer is incorrect");
                        }
                    }
                }
                Statement::PrintStmt { args } => {
                    if args.len() == 1 && args[0] == Expr::Identifier("x_grad".to_string()) {
                        has_likh_x_grad = true;
                    }
                }
                _ => {}
            }
        }

        assert!(has_loss_grad, "Missing loss_grad");
        assert!(has_y_grad, "Missing y_grad");
        assert!(has_x_grad, "Missing x_grad");
        assert!(has_likh_x_grad, "Missing print statement for x_grad");
    }

    #[test]
    fn test_autodiff_matrix_matmul() {
        let src = r#"
            let W = tensor::randn([64, 32], requires_grad=true);
            let X = tensor::randn([1, 64]);
            let Y_target = tensor::ones([1, 32]);
            let Y_pred = X @ W;
            let loss = mean((Y_pred - Y_target) ^ 2);
            loss.backward();
            likh(W.grad);
        "#;
        let tokens = lex(src).unwrap();
        let program = parse(&tokens).unwrap();
        let result = generate_backward(&program).unwrap();

        // Print generated statements for debugging
        for (i, stmt) in result.statements.iter().enumerate() {
            println!("{}: {:?}", i, stmt);
        }

        let mut has_w_grad = false;
        for stmt in &result.statements {
            if let Statement::VarDecl { name, initializer, .. } = stmt {
                if name == "W_grad" {
                    has_w_grad = true;
                    // W_grad = X' @ Y_pred_grad
                    if let Expr::BinaryOp { left, op, right } = initializer {
                        assert_eq!(*op, BinaryOpKind::MatMul);
                        // Left is X transpose
                        if let Expr::UnaryOp { op: UnaryOpKind::Transpose, expr } = &**left {
                            assert_eq!(**expr, Expr::Identifier("X".to_string()));
                        } else {
                            panic!("W_grad left operand is not X'");
                        }
                        assert_eq!(**right, Expr::Identifier("Y_pred_grad".to_string()));
                    } else {
                        panic!("W_grad initializer is not matmul");
                    }
                }
            }
        }
        assert!(has_w_grad, "Missing W_grad");
    }
}
