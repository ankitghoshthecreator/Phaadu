// Step 1: pretend we "read" source code
fn main() {
    let a=10;
    println!("a= {}", a);

    let mut b=18;
    println!("b= {}", b);
    b=1;
    println!("after changing b= {}", b);

    let addition=sum(a,b);
    println!("addition= {}", addition);

}
fn sum(x:i32,y:i32)->i32{
    return x+y;
}