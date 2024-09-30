import dataclasses
from typing import Optional, Tuple, cast, Union

from starkware.cairo.lang.compiler.ast.expr import (
    ExprConst,
    ExprDeref,
    Expression,
    ExprOperator,
    ExprReg,
)
from starkware.cairo.lang.compiler.ast.instructions import (
    AddApInstruction,
    AssertEqInstruction,
    CallInstruction,
    CallLabelInstruction,
    DefineWordInstruction,
    InstructionAst,
    JnzInstruction,
    JumpInstruction,
    JumpToLabelInstruction,
    RetInstruction,
)
from starkware.cairo.lang.compiler.const_expr_checker import is_const_expr
from starkware.cairo.lang.compiler.error_handling import LocationError
from starkware.cairo.lang.compiler.instruction import (
    OFFSET_BITS,
    BytecodeData,
    BytecodeElement,
    M31Instruction,
    Register,
)


class InstructionBuilderError(LocationError):
    pass


def build_instruction(instruction: InstructionAst) -> BytecodeElement:
    if isinstance(instruction.body, AssertEqInstruction):
        return _build_assert_eq_instruction(instruction)
    elif isinstance(instruction.body, JumpInstruction):
        return _build_jump_instruction(instruction)
    elif isinstance(instruction.body, JnzInstruction):
        return _build_jnz_instruction(instruction)
    elif isinstance(instruction.body, CallInstruction):
        return _build_call_instruction(instruction)
    elif isinstance(instruction.body, RetInstruction):
        return _build_ret_instruction(instruction)
    elif isinstance(instruction.body, AddApInstruction):
        return _build_addap_instruction(instruction)
    elif isinstance(instruction.body, DefineWordInstruction):
        return _build_data_word(instruction)
    else:
        raise InstructionBuilderError(
            f"Instructions of type {type(instruction.body).__name__} are not implemented.",
            location=instruction.body.location,
        )


def get_instruction_size(instruction: InstructionAst, allow_auto_deduction: bool = False):
    if allow_auto_deduction:
        # Treat jump to label as a special case, since the label may not have a value yet.
        if isinstance(instruction.body, (JumpToLabelInstruction, CallLabelInstruction)):
            return 2
        if isinstance(instruction.body, AssertEqInstruction):
            try:
                # Try to parse instruction as [reg + off] = const, if it is of that form, its
                # size is 2 even if 'const' can't be resolved yet.
                dst_expr = _parse_dereference(instruction.body.a)
                _parse_register_offset(dst_expr)
                if is_const_expr(instruction.body.b):
                    return 2
            except InstructionBuilderError:
                # If the instruction can't be parsed in that form, try to build it.
                pass
    return build_instruction(instruction).size


def _apply_inverse_syntactic_sugar(instruction_ast: AssertEqInstruction) -> AssertEqInstruction:
    """
    Returns a new instruction AST after applying syntactic sugars.
    Example:
      op0 = dst - op1 => dst = op0 + op1
    """
    if not isinstance(instruction_ast.b, ExprOperator):
        return instruction_ast

    expr: ExprOperator = instruction_ast.b
    for op, inv_op in [("+", "-"), ("*", "/")]:
        if expr.op == inv_op:
            if isinstance(expr.b, ExprConst):
                # The preprocessor should have taken care of this.
                raise InstructionBuilderError(
                    "Subtraction and division are not supported for immediates.",
                    location=expr.b.location,
                )
            return AssertEqInstruction(
                a=expr.a,
                b=ExprOperator(
                    a=instruction_ast.a,
                    op=op,
                    b=expr.b,
                    location=instruction_ast.location,
                ),
                location=instruction_ast.location,
            )

    return instruction_ast


def _build_assert_eq_instruction(instruction_ast: InstructionAst) -> M31Instruction:
    """
    Builds an M31Instruction object from the AST object, assuming the instruction is of type AssertEq.
    If a = b is not a valid Cairo instruction, it will try to write it as b = a. For example,
    "[[ap]] = [fp]" is supported even though its operands are reversed in the bytecode.
    """
    try:
        # First try to parse the instruction in its the original form.
        return _build_assert_eq_instruction_inner(instruction_ast)
    except InstructionBuilderError as exc_:
        # Store the exception in case the second attempt fails as well.
        exc = exc_

    try:
        # If it fails, try to parse it as b = a instead of a = b.
        instruction_body: AssertEqInstruction = cast(AssertEqInstruction, instruction_ast.body)
        return _build_assert_eq_instruction_inner(
            dataclasses.replace(
                instruction_ast,
                body=dataclasses.replace(
                    instruction_body, a=instruction_body.b, b=instruction_body.a
                ),
            )
        )
    except Exception:
        # If both fail, raise the exception thrown by parsing the original form.
        raise exc from None


def _build_assert_eq_instruction_inner(instruction_ast: InstructionAst) -> M31Instruction:
    """
    Builds an M31Instruction object from the AST object, assuming the instruction is of type AssertEq.
    """
    instruction_body: AssertEqInstruction = cast(AssertEqInstruction, instruction_ast.body)

    instruction_body = _apply_inverse_syntactic_sugar(instruction_body)

    dst_expr = _parse_dereference(instruction_body.a)
    dst_register, off0 = _parse_register_offset(dst_expr)

    res_desc = _parse_res(instruction_body.b)

    ap_update = (
        M31Instruction.ApUpdate.ADD1 if instruction_ast.inc_ap else M31Instruction.ApUpdate.REGULAR
    )

    return M31Instruction(
        off0=off0,
        off1=res_desc.off1,
        off2=res_desc.off2,
        imm=res_desc.imm,
        dst_register=dst_register,
        op0_register=res_desc.op0_register,
        op1_addr=res_desc.op1_addr,
        res=res_desc.res,
        pc_update=M31Instruction.PcUpdate.REGULAR,
        ap_update=ap_update,
        fp_update=M31Instruction.FpUpdate.REGULAR,
        opcode=M31Instruction.Opcode.ASSERT_EQ,
    )


def _build_jump_instruction(instruction_ast: InstructionAst) -> M31Instruction:
    """
    Builds an M31Instruction object from the AST object, assuming the instruction is a JumpInstruction.
    """
    instruction_body: JumpInstruction = cast(JumpInstruction, instruction_ast.body)

    res_desc = _parse_res(instruction_body.val)

    ap_update = (
        M31Instruction.ApUpdate.ADD1 if instruction_ast.inc_ap else M31Instruction.ApUpdate.REGULAR
    )
    pc_update = (
        M31Instruction.PcUpdate.JUMP_REL if instruction_body.relative else M31Instruction.PcUpdate.JUMP
    )

    return M31Instruction(
        # In this case dst is not involved. Choose [fp - 1] as the default.
        off0=-1,
        off1=res_desc.off1,
        off2=res_desc.off2,
        imm=res_desc.imm,
        dst_register=Register.FP,
        op0_register=res_desc.op0_register,
        op1_addr=res_desc.op1_addr,
        res=res_desc.res,
        pc_update=pc_update,
        ap_update=ap_update,
        fp_update=M31Instruction.FpUpdate.REGULAR,
        opcode=M31Instruction.Opcode.NOP,
    )


def _build_jnz_instruction(instruction_ast: InstructionAst) -> M31Instruction:
    """
    Builds an M31Instruction object from the AST object, assuming the instruction is a JnzInstruction.
    """
    instruction_body: JnzInstruction = cast(JnzInstruction, instruction_ast.body)

    cond_addr = _parse_dereference(instruction_body.condition)
    dst_register, off0 = _parse_register_offset(cond_addr)

    jump_offset = instruction_body.jump_offset
    if isinstance(jump_offset, ExprDeref):
        op1_reg, off2 = _parse_register_offset(jump_offset.addr)
        imm = None
        op1_addr = M31Instruction.Op1Addr.FP if op1_reg is Register.FP else M31Instruction.Op1Addr.AP
    elif isinstance(jump_offset, ExprConst):
        off2 = 1
        imm = jump_offset.val
        op1_addr = M31Instruction.Op1Addr.IMM
    else:
        raise InstructionBuilderError(
            "Invalid expression for jmp offset.", location=jump_offset.location
        )

    ap_update = (
        M31Instruction.ApUpdate.ADD1 if instruction_ast.inc_ap else M31Instruction.ApUpdate.REGULAR
    )

    return M31Instruction(
        off0=off0,
        # In this case op0 is not involved. Choose[fp - 1] as the default.
        off1=-1,
        off2=off2,
        imm=imm,
        dst_register=dst_register,
        op0_register=Register.FP,
        op1_addr=op1_addr,
        res=M31Instruction.Res.UNCONSTRAINED,
        pc_update=M31Instruction.PcUpdate.JNZ,
        ap_update=ap_update,
        fp_update=M31Instruction.FpUpdate.REGULAR,
        opcode=M31Instruction.Opcode.NOP,
    )


def _build_call_instruction(instruction_ast: InstructionAst) -> M31Instruction:
    """
    Builds an M31Instruction object from the AST object, assuming the instruction is a CallInstruction.
    """
    instruction_body: CallInstruction = cast(CallInstruction, instruction_ast.body)

    val = instruction_body.val
    if isinstance(val, ExprDeref):
        op1_reg, off2 = _parse_register_offset(val.addr)
        imm = None
        op1_addr = M31Instruction.Op1Addr.FP if op1_reg is Register.FP else M31Instruction.Op1Addr.AP
    elif isinstance(val, ExprConst):
        off2 = 1
        imm = val.val
        op1_addr = M31Instruction.Op1Addr.IMM
    else:
        raise InstructionBuilderError("Invalid offset for call.", location=val.location)

    if instruction_ast.inc_ap:
        raise InstructionBuilderError(
            "ap++ may not be used with the call opcode.", location=instruction_ast.location
        )

    pc_update = (
        M31Instruction.PcUpdate.JUMP_REL if instruction_body.relative else M31Instruction.PcUpdate.JUMP
    )

    return M31Instruction(
        # Use dst for [ap] <- fp.
        off0=0,
        # Use op0 for [ap + 1] <- pc.
        off1=1,
        # Use op1 for jmp offset.
        off2=off2,
        imm=imm,
        dst_register=Register.AP,
        op0_register=Register.AP,
        op1_addr=op1_addr,
        res=M31Instruction.Res.OP1,
        pc_update=pc_update,
        ap_update=M31Instruction.ApUpdate.ADD2,
        fp_update=M31Instruction.FpUpdate.AP_PLUS2,
        opcode=M31Instruction.Opcode.CALL,
    )


def _build_ret_instruction(instruction_ast: InstructionAst) -> M31Instruction:
    """
    Builds an M31Instruction object from the AST object, assuming the instruction is a RetInstruction.
    """

    if instruction_ast.inc_ap:
        raise InstructionBuilderError(
            "ap++ may not be used with the ret opcode.", location=instruction_ast.location
        )

    return M31Instruction(
        # Use dst for fp <- [fp - 2].
        off0=-2,
        # In this case op0 is not involved. Choose[fp - 1] as the default.
        off1=-1,
        # Use op1 for pc <- [fp - 1].
        off2=-1,
        imm=None,
        dst_register=Register.FP,
        op0_register=Register.FP,
        op1_addr=M31Instruction.Op1Addr.FP,
        res=M31Instruction.Res.OP1,
        pc_update=M31Instruction.PcUpdate.JUMP,
        ap_update=M31Instruction.ApUpdate.REGULAR,
        fp_update=M31Instruction.FpUpdate.DST,
        opcode=M31Instruction.Opcode.RET,
    )


def _build_addap_instruction(instruction_ast: InstructionAst) -> M31Instruction:
    """
    Builds an M31Instruction object from the AST object, assuming the instruction is an
    AddApInstruction.
    """
    instruction_body: AddApInstruction = cast(AddApInstruction, instruction_ast.body)

    res_desc = _parse_res(instruction_body.expr)

    if instruction_ast.inc_ap:
        raise InstructionBuilderError(
            "ap++ may not be used with the addap opcode.", location=instruction_ast.location
        )

    return M31Instruction(
        # In this case dst is not involved. Choose [fp - 1] as the default.
        off0=-1,
        off1=res_desc.off1,
        off2=res_desc.off2,
        imm=res_desc.imm,
        dst_register=Register.FP,
        op0_register=res_desc.op0_register,
        op1_addr=res_desc.op1_addr,
        res=res_desc.res,
        pc_update=M31Instruction.PcUpdate.REGULAR,
        ap_update=M31Instruction.ApUpdate.ADD,
        fp_update=M31Instruction.FpUpdate.REGULAR,
        opcode=M31Instruction.Opcode.NOP,
    )


def _build_data_word(instruction_ast: InstructionAst) -> BytecodeData:
    """
    Builds a BytecodeData object from the AST object, assuming the instruction is a
    DefineWordInstruction.
    """
    instruction_body: DefineWordInstruction = cast(DefineWordInstruction, instruction_ast.body)

    if instruction_ast.inc_ap:
        raise InstructionBuilderError(
            "ap++ may not be used with the dw opcode.", location=instruction_ast.location
        )

    if not isinstance(instruction_body.expr, ExprConst):
        raise InstructionBuilderError(
            "dw must be followed by a constant expression.", location=instruction_body.expr.location
        )

    return BytecodeData(data=instruction_body.expr.val)


def reg2str(reg: Register) -> str:
    return "fp" if reg is Register.FP else "ap"

"""
Classes that describe the various res operands. Each class has a `suffix_and_nums` method
that returns (`suffix`, `nums`) where `suffix` is a string that describes the operand and
`nums` are its parameters.
"""
@dataclasses.dataclass
class ResImm:
    " = C"
    imm: int
    
    def suffix_and_nums(self):
        return "imm", [self.imm]

@dataclasses.dataclass
class ResDeref:
    " = [reg + off]"
    off: int
    reg: Register

    def suffix_and_nums(self):
        return f"deref_{reg2str(self.reg)}", [self.off]

@dataclasses.dataclass
class ResDoubleDeref:
    " = [[reg + inner] + outer]"
    inner: int
    outer: int
    reg: Register

    def suffix_and_nums(self):
        return f"double_deref_{reg2str(self.reg)}", [self.inner, self.outer]

@dataclasses.dataclass
class ResAdd:
    " = [reg_l + off_l] + [reg_r + off_r]"
    off_l: int
    off_r: int
    reg_l: Register
    reg_r: Register

    def suffix_and_nums(self):
        return f"add_{reg2str(self.reg_l)}_{reg2str(self.reg_r)}", [self.off_l, self.off_r]

@dataclasses.dataclass
class ResMul:
    " = [reg_l + off_l] * [reg_r + off_r]"
    off_l: int
    off_r: int
    reg_l: Register
    reg_r: Register

    def suffix_and_nums(self):
        return f"mul_{reg2str(self.reg_l)}_{reg2str(self.reg_r)}", [self.off_l, self.off_r]

@dataclasses.dataclass
class ResAddImm:
    " = [reg + off] + C"
    off: int
    reg: Register
    imm: int

    def suffix_and_nums(self):
        return f"add_imm_{reg2str(self.reg)}", [self.off, self.imm]

@dataclasses.dataclass
class ResMulImm:
    " = [reg + off] * C"
    off: int
    reg: Register
    imm: int

    def suffix_and_nums(self):
        return f"mul_imm_{reg2str(self.reg)}", [self.off, self.imm]


RES_TYPES = [ResImm, ResDeref, ResDoubleDeref, ResAdd, ResAddImm, ResMul, ResMulImm]

ResDescription = Union[ResImm, ResDeref, ResDoubleDeref, ResAdd, ResAddImm, ResMul, ResMulImm]


def _parse_res(expr: Expression) -> ResDescription:
    """
    Parses the res operand of the instruction and returns the corresponding ResDescription.
    """
    if isinstance(expr, ExprDeref):
        return _parse_res_deref(expr.addr)
    elif isinstance(expr, ExprConst):
        return ResImm(imm=expr.val)
    elif isinstance(expr, ExprOperator):
        return _parse_res_operator(expr)
    else:
        raise InstructionBuilderError("Invalid RHS expression.", location=expr.location)


def _parse_res_deref(expr: Expression) -> ResDescription:
    """
    Given an expression of the form "[reg + off] + off" or "fp + off", returns ResDescription
    corresponding to [[reg + off] + off] or [fp + off] respectively.
    """
    if isinstance(expr, ExprDeref) or (
        isinstance(expr, ExprOperator) and isinstance(expr.a, ExprDeref)
    ):
        # Double dereference.
        inner, off2 = _parse_offset(expr)
        inner_addr = _parse_dereference(inner)
        reg, off1 = _parse_register_offset(inner_addr)
        return ResDoubleDeref(
            inner=off1,
            outer=off2,
            reg=reg,
        )

    # In this case op0 is not involved. Choose [fp - 1] as the default.
    op1_reg, off2 = _parse_register_offset(expr)
    return ResDeref(
        off = off2,
        reg=op1_reg,
    )


def _parse_res_operator(expr: ExprOperator) -> ResDescription:
    """
    Given an expression of the form "[reg + off] * [reg + off]" or "[reg + off] * imm" (* can be
    replaced by +), returns the corresponding ResDescription.
    """
    if expr.op == "+":
        res_types = [ResAdd, ResAddImm]
    elif expr.op == "*":
        res_types = [ResMul, ResMulImm]
    else:
        raise InstructionBuilderError(
            f"Expected '+' or '*', found: '{expr.op}'.", location=expr.location
        )

    # Parse op0.
    op0_expr = _parse_dereference(expr.a)
    op0_register, off1 = _parse_register_offset(op0_expr)

    # Parse op1.
    op1_expr = expr.b
    if isinstance(op1_expr, ExprConst):
        return res_types[1](off=off1, reg=op0_register, imm=op1_expr.val)
    elif isinstance(op1_expr, ExprDeref):
        op1_reg, off2 = _parse_register_offset(op1_expr.addr)
        return res_types[1](off_l=off1, off_r=off2, reg_l=op0_register, reg_r=op1_reg)
    else:
        raise InstructionBuilderError(
            "Expected a constant expression or a dereference expression.",
            location=op1_expr.location,
        )


def _parse_dereference(expr: Expression):
    """
    Given an expression which should be of the form [expr] returns the inner expression.
    Throws an exception if the expr is not of the expected form.
    """

    if not isinstance(expr, ExprDeref):
        raise InstructionBuilderError("Expected a dereference expression.", location=expr.location)
    return expr.addr


def _parse_register_offset(expr: Expression):
    """
    Given an expression of the form "reg", "reg + off" or "reg - off", returns the register and the
    offset.
    """

    register_expr, offset = _parse_offset(expr)
    reg = _parse_register(register_expr)
    return reg, offset


def _parse_offset(expr: Expression) -> Tuple[Expression, int]:
    """
    Given an expression of the form (val + off) or val, where off is in the range [-2**15, 2**15),
    returns val (as an Expression) and the offset.
    Note that if expr is of type ExprOperator, but it doesn't have the expected form, an exception
    will be raised.
    """

    if not isinstance(expr, ExprOperator):
        return expr, 0

    if expr.op == "+":
        sign = 1
    elif expr.op == "-":
        sign = -1
    else:
        raise InstructionBuilderError(
            f"Expected '+' or '-', found: '{expr.op}'.", location=expr.location
        )
    offset_limit = 2 ** (OFFSET_BITS - 1)
    if not isinstance(expr.b, ExprConst) or not -offset_limit <= sign * expr.b.val < offset_limit:
        raise InstructionBuilderError(
            f"Expected a constant offset in the range [-2^{OFFSET_BITS - 1}, 2^{OFFSET_BITS - 1}).",
            location=expr.b.location,
        )
    return expr.a, sign * expr.b.val


def _parse_register(expr: Expression) -> Register:
    """
    Given an expression which should be of type ExprReg, returns the register.
    """
    if not isinstance(expr, ExprReg):
        raise InstructionBuilderError(
            f"Expected a register. Found: {expr.format()}.", location=expr.location
        )

    return expr.reg
