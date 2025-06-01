	.file	"cpu_flops.cc"
	.text
	.p2align 4
	.type	main._omp_fn.0, @function
main._omp_fn.0:
.LFB8301:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$448, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	64(%rdi), %rax
	vmovaps	32(%rdi), %ymm0
	movq	%fs:40, %rdx
	movq	%rdx, 440(%rsp)
	xorl	%edx, %edx
	vmovaps	(%rdi), %ymm1
	leaq	120(%rsp), %rsi
	movq	%rax, 96(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, %rdi
	movq	%rsi, 16(%rsp)
	movq	%rax, 8(%rsp)
	vmovaps	%ymm0, 32(%rsp)
	vmovaps	%ymm1, 64(%rsp)
	vzeroupper
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	testb	%al, %al
	je	.L2
	vmovaps	32(%rsp), %ymm0
	vmovaps	64(%rsp), %ymm1
	leaq	160(%rsp), %rbx
	leaq	416(%rsp), %r14
	leaq	128(%rsp), %r13
.L6:
	movq	112(%rsp), %rax
	movq	%rax, 104(%rsp)
	movq	120(%rsp), %rax
	movq	%rax, 24(%rsp)
.L5:
	movq	%rbx, %rdi
	movl	$32, %ecx
	xorl	%eax, %eax
	rep stosq
	leaq	160(%rsp), %rbx
	movq	%rbx, %r12
.L3:
	vpxor	%xmm2, %xmm2, %xmm2
	movq	%r13, %r15
	vmovdqa	%xmm2, 128(%rsp)
	vmovdqa	%xmm2, 144(%rsp)
	.p2align 4,,10
	.p2align 3
.L7:
	vmovaps	%ymm1, 32(%rsp)
	vmovaps	%ymm0, 64(%rsp)
	vzeroupper
	addq	$4, %r15
	call	rand@PLT
	vxorps	%xmm3, %xmm3, %xmm3
	vmovss	.LC2(%rip), %xmm4
	vmovaps	64(%rsp), %ymm0
	vcvtsi2ssl	%eax, %xmm3, %xmm2
	vmulss	.LC0(%rip), %xmm2, %xmm2
	vmovaps	32(%rsp), %ymm1
	vfmadd132ss	.LC1(%rip), %xmm4, %xmm2
	vmovss	%xmm2, -4(%r15)
	cmpq	%rbx, %r15
	jne	.L7
	vmovaps	128(%rsp), %ymm5
	addq	$32, %r12
	vmovaps	%ymm5, -32(%r12)
	cmpq	%r12, %r14
	jne	.L3
	vmovaps	160(%rsp), %ymm2
	movl	$500000, %eax
	vmovaps	192(%rsp), %ymm9
	vmovaps	224(%rsp), %ymm8
	vmovaps	256(%rsp), %ymm7
	vmovaps	288(%rsp), %ymm6
	vmovaps	320(%rsp), %ymm5
	vmovaps	352(%rsp), %ymm4
	vmovaps	384(%rsp), %ymm3
	.p2align 4,,10
	.p2align 3
.L4:
	vfmadd231ps	%ymm0, %ymm1, %ymm2
	vfmadd231ps	%ymm0, %ymm1, %ymm9
	vfmadd231ps	%ymm0, %ymm1, %ymm8
	vfmadd231ps	%ymm0, %ymm1, %ymm7
	vfmadd231ps	%ymm0, %ymm1, %ymm6
	vfmadd231ps	%ymm0, %ymm1, %ymm5
	vfmadd231ps	%ymm0, %ymm1, %ymm4
	vfmadd231ps	%ymm0, %ymm1, %ymm3
	subq	$1, %rax
	jne	.L4
	vxorps	%xmm10, %xmm10, %xmm10
	movq	104(%rsp), %rax
	movq	96(%rsp), %rdx
	vaddps	%ymm10, %ymm2, %ymm2
	vaddps	%ymm9, %ymm2, %ymm2
	vaddps	%ymm8, %ymm2, %ymm2
	vaddps	%ymm7, %ymm2, %ymm2
	vaddps	%ymm6, %ymm2, %ymm2
	vaddps	%ymm5, %ymm2, %ymm2
	vaddps	%ymm4, %ymm2, %ymm2
	vaddps	%ymm3, %ymm2, %ymm2
	vaddss	(%rdx,%rax,4), %xmm2, %xmm2
	vmovss	%xmm2, (%rdx,%rax,4)
	addq	$1, %rax
	movq	%rax, 104(%rsp)
	cmpq	%rax, 24(%rsp)
	ja	.L5
	movq	16(%rsp), %rsi
	movq	8(%rsp), %rdi
	vmovaps	%ymm1, 32(%rsp)
	vmovaps	%ymm0, 64(%rsp)
	vzeroupper
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	vmovaps	64(%rsp), %ymm0
	vmovaps	32(%rsp), %ymm1
	testb	%al, %al
	jne	.L6
	vzeroupper
.L2:
	call	GOMP_loop_end_nowait@PLT
	movq	440(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L24
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L24:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE8301:
	.size	main._omp_fn.0, .-main._omp_fn.0
	.p2align 4
	.type	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.isra.0, @function
_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.isra.0:
.LFB8303:
	.cfi_startproc
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	movq	%rdi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	testq	%rsi, %rsi
	je	.L28
	movq	%rsi, %rdi
	movq	%rsi, %rbp
	call	strlen@PLT
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	movq	%rbp, %rsi
	movq	%r12, %rdi
	popq	%rbp
	.cfi_def_cfa_offset 16
	movq	%rax, %rdx
	popq	%r12
	.cfi_def_cfa_offset 8
	jmp	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.L28:
	.cfi_restore_state
	movq	(%rdi), %rax
	movq	-24(%rax), %rdi
	addq	%r12, %rdi
	movl	32(%rdi), %esi
	popq	%rax
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	orl	$1, %esi
	jmp	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@PLT
	.cfi_endproc
.LFE8303:
	.size	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.isra.0, .-_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.isra.0
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC7:
	.string	"Avg Wall time        : "
.LC8:
	.string	" s\n"
.LC9:
	.string	"Total FLOPs          : "
.LC11:
	.string	" GFLOP\n"
.LC12:
	.string	"Avg Achieved FLOP    : "
.LC13:
	.string	" GFLOP/s\n"
.LC14:
	.string	"Cycles elapsed       : "
.LC15:
	.string	"\n"
.LC16:
	.string	"Measured CPU freq    : "
.LC17:
	.string	" GHz\n"
.LC18:
	.string	"sum                  : "
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB7770:
	.cfi_startproc
	endbr64
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	vpxor	%xmm0, %xmm0, %xmm0
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	leaq	-80(%rbp), %r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	leaq	-112(%rbp), %rbx
	subq	$256, %rsp
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
	vmovdqa	%xmm0, -112(%rbp)
	vmovdqa	%xmm0, -96(%rbp)
	.p2align 4,,10
	.p2align 3
.L30:
	call	rand@PLT
	vxorps	%xmm2, %xmm2, %xmm2
	addq	$4, %rbx
	vmovss	.LC2(%rip), %xmm3
	vcvtsi2ssl	%eax, %xmm2, %xmm0
	vmulss	.LC0(%rip), %xmm0, %xmm0
	vfmadd132ss	.LC1(%rip), %xmm3, %xmm0
	vmovss	%xmm0, -4(%rbx)
	cmpq	%r12, %rbx
	jne	.L30
	movl	$400000, %edi
	xorl	%r13d, %r13d
	leaq	-212(%rbp), %r14
	call	malloc@PLT
	vmovaps	-112(%rbp), %ymm5
	vxorpd	%xmm1, %xmm1, %xmm1
	leaq	-208(%rbp), %r15
	movq	%rax, -240(%rbp)
	movl	$5, -248(%rbp)
	vmovaps	%ymm5, -304(%rbp)
	vzeroupper
.L31:
	vmovsd	%xmm1, -232(%rbp)
	rdtscp
	movl	%ecx, (%r14)
	salq	$32, %rdx
	movq	%rax, %rbx
	orq	%rdx, %rbx
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	pushq	$0
	movl	$1, %r9d
	xorl	%ecx, %ecx
	pushq	$1
	vmovaps	.LC4(%rip), %ymm4
	movq	%rax, %r12
	xorl	%edx, %edx
	movq	-240(%rbp), %rax
	vmovaps	-304(%rbp), %ymm5
	movl	$100000, %r8d
	movq	%r15, %rsi
	leaq	main._omp_fn.0(%rip), %rdi
	vmovaps	%ymm4, -176(%rbp)
	movq	%rax, -144(%rbp)
	vmovaps	%ymm5, -208(%rbp)
	vzeroupper
	call	GOMP_parallel_loop_nonmonotonic_dynamic@PLT
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	%rax, %rsi
	rdtscp
	vxorpd	%xmm6, %xmm6, %xmm6
	salq	$32, %rdx
	movl	%ecx, (%r14)
	subq	%r12, %rsi
	vmovsd	-232(%rbp), %xmm1
	orq	%rdx, %rax
	vcvtsi2sdq	%rsi, %xmm6, %xmm0
	vdivsd	.LC5(%rip), %xmm0, %xmm0
	subq	%rbx, %rax
	vaddsd	%xmm0, %xmm1, %xmm1
	addq	%rax, %r13
	subl	$1, -248(%rbp)
	popq	%rax
	popq	%rdx
	jne	.L31
	vmovapd	%xmm6, %xmm7
	movq	%r13, %rax
	vdivsd	.LC6(%rip), %xmm1, %xmm4
	vcvtss2sd	-112(%rbp), %xmm6, %xmm1
	vxorpd	%xmm6, %xmm6, %xmm6
	vcvtss2sd	-108(%rbp), %xmm7, %xmm0
	movabsq	$-3689348814741910323, %rdx
	leaq	_ZSt4cout(%rip), %r12
	vaddsd	%xmm6, %xmm1, %xmm1
	leaq	.LC7(%rip), %rsi
	movq	%r12, %rdi
	vmovsd	%xmm4, -232(%rbp)
	mulq	%rdx
	leaq	.LC15(%rip), %r13
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtss2sd	-104(%rbp), %xmm7, %xmm1
	movq	%rdx, %rbx
	movl	$23, %edx
	shrq	$2, %rbx
	vaddsd	%xmm0, %xmm1, %xmm1
	vcvtss2sd	-100(%rbp), %xmm7, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtss2sd	-96(%rbp), %xmm7, %xmm1
	vaddsd	%xmm0, %xmm1, %xmm1
	vcvtss2sd	-92(%rbp), %xmm7, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtss2sd	-88(%rbp), %xmm7, %xmm1
	vaddsd	%xmm0, %xmm1, %xmm1
	vcvtss2sd	-84(%rbp), %xmm7, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm1
	vmovsd	%xmm1, -248(%rbp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	vmovsd	-232(%rbp), %xmm0
	movq	%r12, %rdi
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	leaq	.LC8(%rip), %rsi
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.isra.0
	movl	$23, %edx
	leaq	.LC9(%rip), %rsi
	movq	%r12, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	.LC10(%rip), %rax
	movq	%r12, %rdi
	vmovq	%rax, %xmm0
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	leaq	.LC11(%rip), %rsi
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.isra.0
	movl	$23, %edx
	leaq	.LC12(%rip), %rsi
	movq	%r12, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%r12, %rdi
	vmovsd	.LC10(%rip), %xmm5
	vdivsd	-232(%rbp), %xmm5, %xmm0
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	leaq	.LC13(%rip), %rsi
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.isra.0
	movl	$23, %edx
	leaq	.LC14(%rip), %rsi
	movq	%r12, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	_ZNSo9_M_insertIyEERSoT_@PLT
	movq	%r13, %rsi
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.isra.0
	movl	$23, %edx
	leaq	.LC16(%rip), %rsi
	movq	%r12, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	vxorpd	%xmm7, %xmm7, %xmm7
	movq	%r12, %rdi
	vcvtsi2sdq	%rbx, %xmm7, %xmm0
	vdivsd	-232(%rbp), %xmm0, %xmm0
	vdivsd	.LC5(%rip), %xmm0, %xmm0
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	leaq	.LC17(%rip), %rsi
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.isra.0
	movl	$23, %edx
	leaq	.LC18(%rip), %rsi
	movq	%r12, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	vmovsd	-248(%rbp), %xmm1
	movq	%r12, %rdi
	vmovsd	%xmm1, %xmm1, %xmm0
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	movq	%r13, %rsi
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.isra.0
	movq	-240(%rbp), %rdi
	call	free@PLT
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L36
	leaq	-48(%rbp), %rsp
	xorl	%eax, %eax
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
.L36:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE7770:
	.size	main, .-main
	.p2align 4
	.type	_GLOBAL__sub_I_main, @function
_GLOBAL__sub_I_main:
.LFB8300:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	leaq	_ZStL8__ioinit(%rip), %rbp
	movq	%rbp, %rdi
	call	_ZNSt8ios_base4InitC1Ev@PLT
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	movq	%rbp, %rsi
	popq	%rbp
	.cfi_def_cfa_offset 8
	leaq	__dso_handle(%rip), %rdx
	jmp	__cxa_atexit@PLT
	.cfi_endproc
.LFE8300:
	.size	_GLOBAL__sub_I_main, .-_GLOBAL__sub_I_main
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_main
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC0:
	.long	805306368
	.align 4
.LC1:
	.long	1073741824
	.align 4
.LC2:
	.long	-1082130432
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC4:
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC5:
	.long	0
	.long	1104006501
	.align 8
.LC6:
	.long	0
	.long	1075052544
	.align 8
.LC10:
	.long	0
	.long	1085865984
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
