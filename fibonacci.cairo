%builtins output
func main(output_ptr: felt*) -> (output_ptr: felt*) {
    alloc_locals;
    
    // Load fibonacci_claim_index and copy it to the output segment.
    local fibonacci_claim_index;
    local i;
    local u;
    %{
        ids.fibonacci_claim_index = program_input['fibonacci_claim_index']
    %}
    assert output_ptr[0] = fibonacci_claim_index;

    %{
        from starkware.cairo.lang.vm.relocatable import QM31
        ids.i = QM31.from_ints(0, 1, 0, 0)
        ids.u = QM31.from_ints(0, 0, 1, 0)
    %}
    assert i * i = -1;
    assert u * u = i + 2;

    let res = fib(1, 1, fibonacci_claim_index);
    assert output_ptr[1] = res;

    // Return the updated output_ptr.
    return (output_ptr=&output_ptr[2]);
}

func fib(first_element: felt, second_element: felt, n: felt) -> felt {
    if (n == 0) {
        return second_element;
    }

    return fib(
        first_element=second_element, second_element=first_element * first_element + second_element * second_element, n=n - 1
    );
}

