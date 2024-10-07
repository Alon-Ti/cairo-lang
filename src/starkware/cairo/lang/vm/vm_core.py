import copy
import dataclasses
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from starkware.cairo.lang.compiler.encode import decode_instruction
from starkware.cairo.lang.compiler.instruction import M31Instruction, Register
from starkware.cairo.lang.compiler.program import ProgramBase
from starkware.cairo.lang.vm.builtin_runner import BuiltinRunner
from starkware.cairo.lang.vm.memory_dict import MemoryDict
from starkware.cairo.lang.vm.relocatable import MaybeRelocatable, RelocatableValue
from starkware.cairo.lang.vm.trace_entry import TraceEntry
from starkware.cairo.lang.vm.virtual_machine_base import RunContextBase, VirtualMachineBase
from starkware.cairo.lang.vm.vm_exceptions import PureValueError
from starkware.python.math_utils import div_mod
from starkware.cairo.lang.vm.relocatable import QM31


@dataclasses.dataclass
class Operands:
    """
    Values of the operands.
    """

    dst: MaybeRelocatable
    res: Optional[MaybeRelocatable]
    op0: MaybeRelocatable
    op1: MaybeRelocatable


@dataclasses.dataclass
class RunContext(RunContextBase):
    """
    Contains a complete state of the virtual machine. This includes registers and memory.
    """

    memory: MemoryDict
    pc: MaybeRelocatable
    ap: MaybeRelocatable
    fp: MaybeRelocatable
    prime: int

    def get_instruction_encoding(self) -> QM31:
        """
        Returns the encoded instruction (the value at pc) and the immediate value (the value at
        pc + 1, if it exists in the memory).
        """
        instruction_encoding = self.memory[self.pc]

        assert isinstance(
            instruction_encoding, QM31
        ), f"Instruction should be a QM31. Found: {instruction_encoding}"

        return instruction_encoding

    def compute_dst_addr(self, instruction: M31Instruction):
        base_addr: MaybeRelocatable
        if instruction.dst_register is Register.AP:
            base_addr = self.ap
        elif instruction.dst_register is Register.FP:
            base_addr = self.fp
        else:
            raise NotImplementedError("Invalid dst_register value")
        return (base_addr + instruction.off0) % self.prime

    def compute_op0_addr(self, instruction: M31Instruction):
        base_addr: MaybeRelocatable
        if instruction.op0_register is Register.AP:
            base_addr = self.ap
        elif instruction.op0_register is Register.FP:
            base_addr = self.fp
        else:
            raise NotImplementedError("Invalid op0_register value")
        return (base_addr + instruction.off1) % self.prime

    def compute_op1_addr(self, instruction: M31Instruction, op0: Optional[MaybeRelocatable]):
        base_addr: MaybeRelocatable
        if instruction.op1_addr is M31Instruction.Op1Addr.FP:
            base_addr = self.fp
        elif instruction.op1_addr is M31Instruction.Op1Addr.AP:
            base_addr = self.ap
        elif instruction.op1_addr is M31Instruction.Op1Addr.IMM:
            assert instruction.off2 == 1, "In immediate mode, off2 should be 1."
            base_addr = self.pc
        elif instruction.op1_addr is M31Instruction.Op1Addr.OP0:
            assert op0 is not None, "op0 must be known in double dereference."
            base_addr = op0
        else:
            raise NotImplementedError("Invalid op1_register value")
        return (base_addr + instruction.off2) % self.prime


class VirtualMachine(VirtualMachineBase):
    run_context: RunContext

    def __init__(
        self,
        program: ProgramBase,
        run_context: RunContext,
        hint_locals: Dict[str, Any],
        static_locals: Optional[Dict[str, Any]] = None,
        builtin_runners: Optional[Dict[str, BuiltinRunner]] = None,
        program_base: Optional[MaybeRelocatable] = None,
        enable_instruction_trace: bool = True,
    ):
        """
        See documentation in VirtualMachineBase.

        program_base - The pc of the first instruction in program (default is run_context.pc).
        """
        self.run_context = copy.copy(run_context)  # Shallow copy.
        if program_base is None:
            program_base = run_context.pc
        if builtin_runners is None:
            builtin_runners = {}

        super().__init__(
            program=program,
            run_context=self.run_context,
            hint_locals=hint_locals,
            static_locals=static_locals,
            builtin_runners=builtin_runners,
            program_base=program_base,
        )

        # A set to track the memory addresses accessed by actual Cairo instructions (as opposed to
        # hints), necessary for accurate counting of memory holes.
        self.accessed_addresses: Set[MaybeRelocatable] = {
            program_base + i for i in range(len(self.program.data))
        }

        self._trace: Optional[List[TraceEntry[MaybeRelocatable]]] = (
            [] if enable_instruction_trace else None
        )

        # Current step.
        self.current_step = 0

        # This flag can be set to true by hints to avoid the execution of the current step in
        # step() (so that only the hint will be performed, but nothing else will happen).
        self.skip_instruction_execution = False

        # Set this flag to False to avoid tracking register values each instruction.
        self.enable_instruction_trace = enable_instruction_trace

    @property
    def trace(self) -> List[TraceEntry[MaybeRelocatable]]:
        assert self._trace is not None, "Trace is disabled."
        return self._trace

    def update_registers(self, instruction: M31Instruction, operands: Operands):
        # Update fp.
        if instruction.fp_update is M31Instruction.FpUpdate.AP_PLUS2:
            self.run_context.fp = self.run_context.ap + 2
        elif instruction.fp_update is M31Instruction.FpUpdate.DST:
            self.run_context.fp = operands.dst
        elif instruction.fp_update is not M31Instruction.FpUpdate.REGULAR:
            raise NotImplementedError("Invalid fp_update value")

        # Update ap.
        if instruction.ap_update is M31Instruction.ApUpdate.ADD:
            if operands.res is None:
                raise NotImplementedError("Res.UNCONSTRAINED cannot be used with ApUpdate.ADD")
            self.run_context.ap += operands.res % self.prime
        elif instruction.ap_update is M31Instruction.ApUpdate.ADD1:
            self.run_context.ap += 1
        elif instruction.ap_update is M31Instruction.ApUpdate.ADD2:
            self.run_context.ap += 2
        elif instruction.ap_update is not M31Instruction.ApUpdate.REGULAR:
            raise NotImplementedError("Invalid ap_update value")
        self.run_context.ap = self.run_context.ap % self.prime

        # Update pc.
        # The pc update should be done last so that we will have the correct pc in case of an
        # exception during one of the updates above.
        if instruction.pc_update is M31Instruction.PcUpdate.REGULAR:
            self.run_context.pc += instruction.size
        elif instruction.pc_update is M31Instruction.PcUpdate.JUMP:
            if operands.res is None:
                raise NotImplementedError("Res.UNCONSTRAINED cannot be used with PcUpdate.JUMP")
            self.run_context.pc = operands.res
        elif instruction.pc_update is M31Instruction.PcUpdate.JUMP_REL:
            if operands.res is None:
                raise NotImplementedError("Res.UNCONSTRAINED cannot be used with PcUpdate.JUMP_REL")
            if not isinstance(operands.res, int):
                raise PureValueError("jmp rel", operands.res)
            self.run_context.pc += operands.res
        elif instruction.pc_update is M31Instruction.PcUpdate.JNZ:
            if self.is_zero(operands.dst):
                self.run_context.pc += instruction.size
            else:
                self.run_context.pc += operands.op1
        else:
            raise NotImplementedError("Invalid pc_update value")
        self.run_context.pc = self.run_context.pc % self.prime

    def deduce_op0(
        self,
        instruction: M31Instruction,
        dst: Optional[MaybeRelocatable],
        op1: Optional[MaybeRelocatable],
    ) -> Tuple[Optional[MaybeRelocatable], Optional[MaybeRelocatable]]:
        """
        Returns a tuple (deduced_op0, deduced_res).
        Deduces the value of op0 if possible (based on dst and op1). Otherwise, returns None.
        If res was already deduced, returns its deduced value as well.
        """
        if instruction.opcode is M31Instruction.Opcode.CALL:
            return self.run_context.pc + instruction.size, None
        elif instruction.opcode is M31Instruction.Opcode.ASSERT_EQ:
            if (instruction.res is M31Instruction.Res.ADD) and (dst is not None) and (op1 is not None):
                return (dst - op1) % self.prime, dst  # type: ignore
            elif (
                (instruction.res is M31Instruction.Res.MUL)
                and isinstance(dst, int)
                and isinstance(op1, int)
            ):
                assert (
                    op1 != 0
                ), f"Cannot deduce operand in '0 = ? * {dst}' (possibly due to division by 0)."

                return div_mod(dst, op1, self.prime), dst
        return None, None

    def deduce_op1(
        self,
        instruction: M31Instruction,
        dst: Optional[MaybeRelocatable],
        op0: Optional[MaybeRelocatable],
    ) -> Tuple[Optional[MaybeRelocatable], Optional[MaybeRelocatable]]:
        """
        Returns a tuple (deduced_op1, deduced_res).
        Deduces the value of op1 if possible (based on dst and op0). Otherwise, returns None.
        If res was already deduced, returns its deduced value as well.
        """
        if instruction.opcode is M31Instruction.Opcode.ASSERT_EQ:
            if (instruction.res is M31Instruction.Res.OP1) and (dst is not None):
                return dst, dst
            elif (
                (instruction.res is M31Instruction.Res.ADD) and (dst is not None) and (op0 is not None)
            ):
                return (dst - op0) % self.prime, dst  # type: ignore
            elif (
                (instruction.res is M31Instruction.Res.MUL)
                and isinstance(dst, int)
                and isinstance(op0, int)
                and op0 != 0
            ):
                return div_mod(dst, op0, self.prime), dst
        return None, None

    def compute_res(
        self,
        instruction: M31Instruction,
        op0: MaybeRelocatable,
        op1: MaybeRelocatable,
    ) -> Optional[MaybeRelocatable]:
        """
        Computes the value of res if possible.
        """
        if instruction.res is M31Instruction.Res.OP1:
            return op1
        elif instruction.res is M31Instruction.Res.ADD:
            return (op0 + op1) % self.prime
        elif instruction.res is M31Instruction.Res.MUL:
            if isinstance(op0, RelocatableValue) or isinstance(op1, RelocatableValue):
                raise PureValueError("*", op0, op1)
            return (op0 * op1) % self.prime
        elif instruction.res is M31Instruction.Res.UNCONSTRAINED:
            # In this case res should be the inverse of dst.
            # For efficiency, we do not compute it here.
            return None
        else:
            raise NotImplementedError("Invalid res value")

    def compute_operands(self, instruction: M31Instruction) -> Tuple[Operands, List[int]]:
        """
        Computes the values of the operands. Deduces dst if needed.
        Returns:
          operands - an Operands instance with the values of the operands.
          mem_addresses - the memory addresses for the 3 memory units used (dst, op0, op1).
        """
        # Try to fetch dst, op0, op1.
        # op0 throughout this function represents the value at op0_addr.
        # If op0 is set, this implies that we are going to set memory at op0_addr to that value.
        # Same for op1, dst.
        dst_addr = self.run_context.compute_dst_addr(instruction)
        dst: Optional[MaybeRelocatable] = self.validated_memory.get(dst_addr)
        op0_addr = self.run_context.compute_op0_addr(instruction)
        op0: Optional[MaybeRelocatable] = self.validated_memory.get(op0_addr)
        op1_addr = self.run_context.compute_op1_addr(instruction, op0=op0)
        op1: Optional[MaybeRelocatable] = self.validated_memory.get(op1_addr)
        # res throughout this function represents the computation on op0,op1
        # as defined in decode.py.
        # If it is set, this implies that compute_res(...) will return this value.
        # If it is set without invoking compute_res(), this is an optimization, but should not
        # yield a different result.
        # In particular, res may be different than dst, even in ASSERT_EQ. In this case,
        # The ASSERT_EQ validation will fail in opcode_assertions().
        res: Optional[MaybeRelocatable] = None

        # Auto deduction rules.
        # Note: This may fail to deduce if 2 auto deduction rules are needed to be used in
        # a different order.
        if op0 is None:
            op0 = self.deduce_memory_cell(op0_addr)
        if op1 is None:
            op1 = self.deduce_memory_cell(op1_addr)

        should_update_dst = dst is None
        should_update_op0 = op0 is None
        should_update_op1 = op1 is None

        # Deduce op0 if needed.
        if op0 is None:
            op0, deduced_res = self.deduce_op0(instruction, dst, op1)
            if res is None:
                res = deduced_res

        # Deduce op1 if needed.
        if op1 is None:
            op1, deduced_res = self.deduce_op1(instruction, dst, op0)
            if res is None:
                res = deduced_res

        # Force pulling op0, op1 from memory for soundness test
        # and to get an informative error message if they were not computed.
        if op0 is None:
            op0 = self.validated_memory[op0_addr]
        if op1 is None:
            op1 = self.validated_memory[op1_addr]

        # Compute res if needed.
        if res is None:
            res = self.compute_res(instruction, op0, op1)

        # Deduce dst.
        if dst is None:
            if instruction.opcode is M31Instruction.Opcode.ASSERT_EQ and res is not None:
                dst = res
            elif instruction.opcode is M31Instruction.Opcode.CALL:
                dst = self.run_context.fp

        # Force pulling dst from memory for soundness.
        if dst is None:
            dst = self.validated_memory[dst_addr]

        # Write updated values.
        if should_update_dst:
            self.validated_memory[dst_addr] = dst
        if should_update_op0:
            self.validated_memory[op0_addr] = op0
        if should_update_op1:
            self.validated_memory[op1_addr] = op1

        return (
            Operands(dst=dst, op0=op0, op1=op1, res=res),
            [dst_addr, op0_addr, op1_addr],
        )

    def is_zero(self, value):
        """
        Returns True if value is zero (used for jnz instructions).
        This function can be overridden by subclasses.
        """
        if isinstance(value, int):
            return value == 0

        if isinstance(value, RelocatableValue) and value.offset >= 0:
            return False
        raise PureValueError("jmp != 0", value)

    def is_integer_value(self, value):
        """
        Returns True if value is integer rather than relocatable.
        This function can be overridden by subclasses.
        """
        return isinstance(value, int)

    @staticmethod
    @lru_cache(None)
    def decode_instruction(encoded_inst: QM31) -> M31Instruction:
        return decode_instruction(encoded_inst.to_tuple())

    def decode_current_instruction(self) -> M31Instruction:
        try:
            instruction_encoding = self.run_context.get_instruction_encoding()
            instruction = self.decode_instruction(instruction_encoding)
        except Exception as exc:
            raise self.as_vm_exception(exc) from None

        return instruction

    def opcode_assertions(self, instruction: M31Instruction, operands: Operands):
        if instruction.opcode is M31Instruction.Opcode.ASSERT_EQ:
            if operands.res is None:
                raise NotImplementedError("Res.UNCONSTRAINED cannot be used with Opcode.ASSERT_EQ")
            if operands.dst != operands.res and not self.check_eq(operands.dst, operands.res):
                raise Exception(
                    f"An ASSERT_EQ instruction failed: {operands.dst} != {operands.res}."
                )
        elif instruction.opcode is M31Instruction.Opcode.CALL:
            return_pc = self.run_context.pc + instruction.size
            if operands.op0 != return_pc and not self.check_eq(operands.op0, return_pc):
                raise Exception(
                    "Call failed to write return-pc (inconsistent op0): "
                    + f"{operands.op0} != {return_pc}. Did you forget to increment ap?"
                )
            return_fp = self.run_context.fp
            if operands.dst != return_fp and not self.check_eq(operands.dst, return_fp):
                raise Exception(
                    "Call failed to write return-fp (inconsistent dst): "
                    + f"{operands.dst} != {return_fp}. Did you forget to increment ap?"
                )
        elif instruction.opcode in [M31Instruction.Opcode.RET, M31Instruction.Opcode.NOP]:
            # Nothing to check.
            pass
        else:
            raise NotImplementedError(f"Unsupported opcode {instruction.opcode}.")

    def run_instruction(self, instruction):
        try: 
            if instruction.opcode.startswith("addap"):
                self.run_addap_instruction(instruction)
            elif instruction.opcode.startswith("assert"):
                self.run_assert_instruction(instruction)
            elif instruction.opcode.startswith("call"):
                self.run_call_instruction(instruction)
            elif instruction.opcode.startswith("jmp"):
                self.run_jmp_instruction(instruction)
            elif instruction.opcode.startswith("jnz"):
                self.run_jnz_instruction(instruction)
            elif instruction.opcode == "ret":
                self.run_ret_instruction()
            else:
                assert False, "Unsupported opcode {instruction.opcode}."
        except Exception as exc:
            raise self.as_vm_exception(exc) from None

        # Write to trace.
        if self.enable_instruction_trace:
            self.trace.append(
                TraceEntry(
                    pc=self.run_context.pc,
                    ap=self.run_context.ap,
                    fp=self.run_context.fp,
                )
            )

        self.accessed_addresses.add(self.run_context.pc)

        self.current_step += 1

    def run_addap_instruction(self, instruction: M31Instruction):
        assert instruction.opcode == "addap_imm", f"Unsupported instruction: {instruction.opcode}"

        imm = instruction.operands[0]
        self.run_context.ap += imm
        self.run_context.pc += 1

    def addr_and_val(self, op: str, off: int) -> Tuple[Optional[MaybeRelocatable], Optional[MaybeRelocatable]]:
        if off >= 2**16:
            off -= self.prime
        if op == "imm":
            return None, off
        addr = None
        if op == "ap":
            addr = self.run_context.ap + off
        if op == "fp":
            addr = self.run_context.fp + off
        self.accessed_addresses.add(addr)
        return addr, self.validated_memory.get(addr)

    def run_assert_instruction(self, instruction: M31Instruction):
        opcode = instruction.opcode[len("assert_"):]
        appp = opcode.endswith("appp")
        if appp:
            opcode = opcode[:-len("_appp")]

        try:
            for op0 in ["ap", "fp"]:
                for op1 in ["ap", "fp", "imm"]:
                    for op2 in ["ap", "fp"]:
                        if opcode == f"{op0}_add_{op1}_{op2}":
                            return self.run_add(op0, op1, op2, instruction.operands)
                        if opcode == f"{op0}_mul_{op1}_{op2}":
                            return self.run_mul(op0, op1, op2, instruction.operands)
                        if opcode == f"{op0}_imm":
                            return self.run_assign_immediate(op0, instruction.operands)
                        if opcode == f"{op0}_deref_{op1}":
                            return self.run_equate(op0, op1, instruction.operands)
                        if opcode == f"{op0}_double_deref_{op1}":
                            return self.run_double_deref(op0, op1, instruction.operands)
        finally:
            self.run_context.pc += 1
            if appp:
                self.run_context.ap += 1

        assert False, f"Unsupported instruction: {instruction.opcode}"

    def run_add(self, op0: str, op1: str, op2: str, operands: List[int]):
        addr0, val0 = self.addr_and_val(op0, operands[0])
        addr1, val1 = self.addr_and_val(op1, operands[1])
        addr2, val2 = self.addr_and_val(op2, operands[2])

        if val0 is not  None and val1 is not  None and val2 is not  None:
            assert val0 == val1 + val2, f"Assertion failed: {val0} != {val1} + {val2}"

        if val0 is None:
            assert val1 is not None and val2 is not None, "Cannot deduce more than one operand"
            self.validated_memory[addr0] = val1 + val2

        if val1 is None:
            assert val0 is not None and val2 is not None, "Cannot deduce more than one operand"
            self.validated_memory[addr1] = val2 - val0

        if val2 is None:
            assert val1 is not None and val0 is not None, "Cannot deduce more than one operand"
            self.validated_memory[addr2] = val1 - val0

    def run_mul(self, op0: str, op1: str, op2: str, operands: List[int]):
        addr0, val0 = self.addr_and_val(op0, operands[0])
        addr1, val1 = self.addr_and_val(op1, operands[1])
        addr2, val2 = self.addr_and_val(op2, operands[2])

        if val0 is not  None and val1 is not  None and val2 is not  None:
            assert val0 == val1 * val2, f"Assertion failed: {val0} != {val1} + {val2}"

        if val0 is None:
            assert val1 is not None and val2 is not None, "Cannot deduce more than one operand"
            self.validated_memory[addr0] = val1 * val2

        if val1 is None:
            assert val0 is not None and val2 is not None, "Cannot deduce more than one operand"
            self.validated_memory[addr1] = val2 / val0

        if val2 is None:
            assert val1 is not None and val0 is not None, "Cannot deduce more than one operand"
            self.validated_memory[addr2] = val1 / val0

    def run_assign_immediate(self, op0: str, operands: List[int]):
        addr0, val0 = self.addr_and_val(op0, operands[0])
        if val0 is not None:
            assert val0 == operands[1], f"Assertion failed: {val0} != {operands[1]}"
        if val0 is None:
            self.validated_memory[addr0] = operands[1]

    def run_equate(self, op0: str, op1: str, operands: List[int]):
        addr0, val0 = self.addr_and_val(op0, operands[0])
        addr1, val1 = self.addr_and_val(op1, operands[1])
        if val0 is not None and val1 is not None:
            assert val0 == val1, f"Assertion failed: {val0} != {val1}"
        if val0 is None:
            assert val1 is not None, "Cannot deduce more than one operand"
            self.validated_memory[addr0] = val1
        if val1 is None:
            assert val0 is not None, "Cannot deduce more than one operand"
            self.validated_memory[addr1] = val0

    def run_double_deref(self, op0: str, op1: str, operands: List[int]):
        addr0, val0 = self.addr_and_val(op0, operands[0])
        _, outer_addr = self.addr_and_val(op1, operands[1])
        assert outer_addr is not None, "Address cannot be deduced for double dereference"

        outer_off = operands[2]
        if outer_off >= 2**16:
            outer_off -= self.prime

        outer_addr = outer_addr + outer_off

        self.accessed_addresses.add(outer_addr)
        outer_val = self.validated_memory.get(outer_addr)

        if val0 is not None and outer_val is not None:
            assert val0 == outer_val, f"Assertion failed: {val0} != {outer_val}"

        if val0 is None:
            assert outer_val is not None, "Cannot deduce more than one operand"
            self.validated_memory[addr0] = outer_val

        if outer_val is None:
            assert val0 is not None, "Cannot deduce more than one operand"
            self.validated_memory[outer_addr] = val0

    def run_call_instruction(self, instruction: M31Instruction):
        opcode = instruction.opcode[len("call_"):]

        base_addr = QM31.from_int(0)
        if opcode.startswith("rel"):
            base_addr = self.run_context.pc
        else:
            assert opcode.startswith("abs"), f"Unsupported instruction: {instruction.opcode}"

        opcode = opcode[len("abs_"):]

        _, val = self.addr_and_val(opcode, instruction.operands[0])
        assert val is not None, "Address cannot be deduced for call"

        self.validated_memory[self.run_context.ap] = self.run_context.fp
        self.validated_memory[self.run_context.ap + 1] = self.run_context.pc + 1

        self.accessed_addresses.add(self.run_context.ap)
        self.accessed_addresses.add(self.run_context.ap + 1)

        self.run_context.pc = base_addr + val
        self.run_context.ap += 2
        self.run_context.fp = self.run_context.ap

    def run_jmp_instruction(self, instruction: M31Instruction):
        opcode = instruction.opcode[len("jmp_"):]

        appp = opcode.endswith("appp")
        if appp:
            opcode = opcode[:-len("_appp")]

        base_addr = QM31.from_int(0)
        if opcode.startswith("rel"):
            base_addr = self.run_context.pc
        else:
            assert opcode.startswith("abs"), f"Unsupported instruction: {instruction.opcode}"

        opcode = opcode[len("abs_"):]

        if opcode == "imm":
            self.run_context.pc = base_addr + instruction.operands[0]
            return
        elif opcode.startswith("deref"):
            opcode = opcode[len("deref_"):]
            _, val = self.addr_and_val(opcode, instruction.operands[0])
            assert val is not None, "Address cannot be deduced for jump"
            self.run_context.pc = base_addr + val
            return
        elif opcode.startswith("double_deref"):
            opcode = opcode[len("double_deref_"):]
            _, outer_addr = self.addr_and_val(opcode, instruction.operands[0])
            assert outer_addr is not None, "Address cannot be deduced for double dereference"
            outer_val = self.validated_memory[outer_addr]
            self.accessed_addresses.add(outer_addr)
            self.run_context.pc = base_addr + outer_val
        else:
            assert False, f"Unsupported instruction: {instruction.opcode}"

        if appp:
            self.run_context.ap += 1


    def run_jnz_instruction(self, instruction: M31Instruction):
        opcode = instruction.opcode[len("jnz_"):]

        appp = opcode.endswith("appp")
        if appp:
            opcode = opcode[:-len("_appp")]

        off_op, cond_op = opcode.split("_")
        assert off_op in ["ap", "fp", "imm"], f"Unsupported opcode: {instruction.opcode}"
        assert cond_op in ["ap", "fp"], f"Unsupported opcode: {instruction.opcode}"

        _, off_val = self.addr_and_val(off_op, instruction.operands[0])
        _, cond_val = self.addr_and_val(cond_op, instruction.operands[1])

        assert off_val is not None, "Offset cannot be deduced for jnz"
        assert cond_val is not None, "Condition cannot be deduced for jnz"

        if appp:
            self.run_context.ap += 1
        if not cond_val.is_zero():
            self.run_context.pc += off_val
        else:
            self.run_context.pc += 1

    def run_ret_instruction(self):
        self.run_context.pc = self.validated_memory[self.run_context.fp - 1]
        self.run_context.fp = self.validated_memory[self.run_context.fp - 2]


    def step(self):
        self.skip_instruction_execution = False
        # Execute hints.
        for hint_index, hint in enumerate(self.hints.get(self.run_context.pc, [])):
            exec_locals = self.exec_scopes[-1]
            exec_locals["memory"] = memory = self.validated_memory
            exec_locals["ap"] = ap = self.run_context.ap
            exec_locals["fp"] = fp = self.run_context.fp
            exec_locals["pc"] = pc = self.run_context.pc
            exec_locals["current_step"] = self.current_step
            exec_locals["ids"] = hint.consts(pc, ap, fp, memory)

            exec_locals["vm_load_program"] = self.load_program
            exec_locals["vm_enter_scope"] = self.enter_scope
            exec_locals["vm_exit_scope"] = self.exit_scope
            exec_locals.update(self.static_locals)
            exec_locals["builtin_runners"] = self.builtin_runners
            exec_locals.update(self.builtin_runners)

            self.exec_hint(hint.compiled, exec_locals, hint_index=hint_index)

            # There are memory leaks in 'exec_scopes'.
            # So, we clear some fields in order to reduce the problem.
            for name in self.builtin_runners:
                del exec_locals[name]

            del exec_locals["builtin_runners"]
            for name in self.static_locals:
                del exec_locals[name]

            del exec_locals["vm_exit_scope"]
            del exec_locals["vm_enter_scope"]
            del exec_locals["vm_load_program"]
            del exec_locals["ids"]
            del exec_locals["memory"]

            if self.skip_instruction_execution:
                return

        # Decode.
        instruction = self.decode_current_instruction()

        # Run.
        self.run_instruction(instruction)
