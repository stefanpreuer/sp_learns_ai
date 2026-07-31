# Learning Rust

## Ownership model

Rust's distinguishing concept is its ownership model. By this every value has exactly one owner, with owners owning none or many values. An owner is responsible for dropping a value once the owner's own lifetime ends.

Ownership in total is therefore a tree reshaping itself along the control flow of a program. At every point along this control flow, ownership must be consistent. Since ownership is checked at compile-time the reshaping of the tree is constrained to what can be statically tracked.

The memory- and concurrency-safety guarantees Rust ensures at compile time are based on this ownership model.

What are the owners of values:

- Program (root)
- Static variables
- Local variables and parameters of functions
- Structs and its individual fields
- Arrays and its elements
- Expressions (temporary values)
- ...

The ownership tree is reshaped by moving values from one owner to another and by dropping values at the end of the lifetime of an owner. All tracked at compile time and enforced by the code generated based on this semantic model.

ownership, reborrow and Non-Lexical Lifetimes (NLL).

### Moving values

Initializing or assigning a value to a new target moves the ownership of it from the source. Targets are e.g. variables, function parameters, struct fields, or array elements. The source gets uninitialized by this and if the target owned value before assignment this old value gets dropped - think of dropping out of the ownership tree by another value taking the same position in it. The source can also be a temporary value of an expression for which assignment is anchoring them in the ownership tree by the target taking over ownership.

There are only special `Copy` types, whose values are actually copied. Think of primitive value types and compositions of them.

### References

The ownership tree is rather rigid since moving and dropping values are the only operations on it. References provide an additional concept, allowing one to borrow a value without taking ownership of it. This allows one to operate on a value on a value for a slice of execution, e.g. during a function call, or anchored in a struct operated on for a while. The crucial thing is that for memory safety the reference must not outlive the referent. Rust ensures this by compile time analysis, tracking lifetimes of values and references.

Furthermore Rust enforces the multiple readers or single modifier rule. This rule is crucial for concurrency safety but it also ensures safety in non-concurrent contexts like preventing unintended self-assignment or -modification. Without references unique ownership automatically guarantees this, since a value can always accessed only by its owner. However, references introduce aliases to a value and therefore additional lifetime tracking is needed.

To avoid unnecessary rigidness the compiler implements *Non-Lexical Lifetime (NLL)* analysis. By the control flow analysis allows tracking lifetimes much better than just by lexical scope. This allows using reborrowing, or temproary dead references.

E.g. NLL tracks lifetime not just by lexical scope, but considers actual control flow and value usage.

```rust
let y;
{
  let x = 32;
  y = &x;
}
// `y` not used after end of lifetime of referent `x`.

// println!(y); // This would would result in a compiler-error though.
```

### Reference counted references

In exceptional cases it might be beneficial to support shared ownership of a value with dynamic lifetime ending when last owner's lifetime ends. Rust provides `Rc` and `Arc` types for this, which implement reference counting to achieve shared ownership. This is similar to what garbage collection does, although dropping does not happen as an asynchronous background activity, but synchronously on last owner's lifetime ending.
