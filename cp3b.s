	.file	"cp3b.cc"
	.text
	.p2align 4
	.type	_ZN6MatrixC2EiiPKf._omp_fn.0, @function
_ZN6MatrixC2EiiPKf._omp_fn.0:
.LFB8708:
	.cfi_startproc
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	movq	(%rdi), %r12
	movl	8(%r12), %ebx
	call	omp_get_num_threads@PLT
	movl	%eax, %ebp
	call	omp_get_thread_num@PLT
	movl	%eax, %ecx
	movl	%ebx, %eax
	imull	%ebx, %eax
	cltd
	idivl	%ebp
	cmpl	%edx, %ecx
	jl	.L2
.L4:
	imull	%eax, %ecx
	addl	%ecx, %edx
	leal	(%rax,%rdx), %ecx
	cmpl	%ecx, %edx
	jge	.L5
	subl	$1, %eax
	movslq	%edx, %rdx
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	xorl	%esi, %esi
	leaq	4(,%rax,4), %r8
	movq	32(%r12), %rax
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	leaq	(%rax,%rdx,4), %rdi
	movq	%r8, %rdx
	jmp	memset@PLT
	.p2align 4,,10
	.p2align 3
.L5:
	.cfi_restore_state
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L2:
	.cfi_restore_state
	addl	$1, %eax
	xorl	%edx, %edx
	jmp	.L4
	.cfi_endproc
.LFE8708:
	.size	_ZN6MatrixC2EiiPKf._omp_fn.0, .-_ZN6MatrixC2EiiPKf._omp_fn.0
	.p2align 4
	.type	_ZN6Matrix24standardize_rows_swizzleEv._omp_fn.0, @function
_ZN6Matrix24standardize_rows_swizzleEv._omp_fn.0:
.LFB8709:
	.cfi_startproc
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
	pushq	%r10
	pushq	%rbx
	subq	$272, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 10, -56
	.cfi_offset 3, -64
	movq	(%rdi), %r10
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
	movq	%r10, -168(%rbp)
	call	omp_get_num_threads@PLT
	movl	%eax, %ebx
	call	omp_get_thread_num@PLT
	movq	-168(%rbp), %r10
	movl	%eax, %ecx
	movl	8(%r10), %edx
	leal	14(%rdx), %eax
	addl	$7, %edx
	cmovns	%edx, %eax
	sarl	$3, %eax
	cltd
	idivl	%ebx
	cmpl	%edx, %ecx
	jl	.L9
.L59:
	imull	%eax, %ecx
	addl	%ecx, %edx
	addl	%edx, %eax
	cmpl	%eax, %edx
	jge	.L8
	movl	(%r10), %r9d
	sall	$3, %eax
	movslq	4(%r10), %r8
	vxorps	%xmm1, %xmm1, %xmm1
	leal	0(,%rdx,8), %r11d
	movl	%eax, -288(%rbp)
	cmpl	%r9d, %r11d
	jge	.L11
	cmpl	%r9d, %eax
	movl	%r9d, %ebx
	vxorpd	%xmm6, %xmm6, %xmm6
	movl	%r11d, %r13d
	cmovg	%r9d, %eax
	vxorps	%xmm5, %xmm5, %xmm5
	movq	%r10, %r14
	movl	%r8d, %r12d
	movl	%eax, -292(%rbp)
	leal	0(,%r8,8), %eax
	movl	%eax, -296(%rbp)
	leal	8(%r11), %eax
	movl	%eax, -264(%rbp)
	imull	%r8d, %eax
	movl	%eax, %r15d
	movl	%r8d, %eax
	imull	%r11d, %eax
	movl	%r15d, %r9d
	movl	%eax, -276(%rbp)
	imull	$-7, %r8d, %eax
	movl	%eax, -308(%rbp)
	xorl	%eax, %eax
	testl	%r8d, %r8d
	cmovns	%r8, %rax
	movl	%eax, -280(%rbp)
	movq	%rax, -304(%rbp)
	.p2align 4,,10
	.p2align 3
.L14:
	vpxor	%xmm0, %xmm0, %xmm0
	leal	-1(%r12), %r15d
	movl	-276(%rbp), %esi
	xorl	%ecx, %ecx
	vmovdqa	%xmm0, -128(%rbp)
	addq	$2, %r15
	vmovdqa	%xmm0, -112(%rbp)
	vmovdqa	%xmm0, -96(%rbp)
	vmovdqa	%xmm0, -80(%rbp)
	vmovdqa	%xmm0, -160(%rbp)
	vmovdqa	%xmm0, -144(%rbp)
	.p2align 4,,10
	.p2align 3
.L12:
	leal	0(%r13,%rcx), %eax
	cmpl	%ebx, %eax
	jge	.L56
	leaq	-160(%rbp), %rdi
	vmovss	(%rdi,%rcx,4), %xmm4
	testl	%r12d, %r12d
	jle	.L58
	movq	16(%r14), %rdx
	leaq	-128(%rbp), %r8
	movslq	%esi, %rax
	vmovsd	(%r8,%rcx,8), %xmm7
	leaq	(%rdx,%rax,4), %rdx
	movl	$1, %eax
	.p2align 4,,10
	.p2align 3
.L57:
	vcvtsi2sdl	%eax, %xmm1, %xmm2
	vcvtss2sd	-4(%rdx,%rax,4), %xmm1, %xmm0
	vsubsd	%xmm7, %xmm0, %xmm3
	vcvtss2sd	%xmm4, %xmm4, %xmm4
	addq	$1, %rax
	vdivsd	%xmm2, %xmm3, %xmm2
	vaddsd	%xmm2, %xmm7, %xmm7
	vsubsd	%xmm7, %xmm0, %xmm0
	vfmadd231sd	%xmm0, %xmm3, %xmm4
	vcvtsd2ss	%xmm4, %xmm4, %xmm4
	cmpq	%r15, %rax
	jne	.L57
	vmovsd	%xmm7, (%r8,%rcx,8)
.L58:
	vcvtss2sd	%xmm4, %xmm4, %xmm4
	vucomisd	%xmm4, %xmm6
	ja	.L86
	vsqrtsd	%xmm4, %xmm4, %xmm4
	vmovsd	.LC2(%rip), %xmm0
	addl	%r12d, %esi
	vdivsd	%xmm4, %xmm0, %xmm4
	vcvtsd2ss	%xmm4, %xmm4, %xmm4
	vmovss	%xmm4, (%rdi,%rcx,4)
	addq	$1, %rcx
	cmpq	$8, %rcx
	jne	.L12
.L56:
	movl	12(%r14), %eax
	movl	%eax, -272(%rbp)
	testl	%r12d, %r12d
	jle	.L46
	vmovsd	-112(%rbp), %xmm3
	leal	7(%r13), %edi
	movl	%r13d, %ecx
	movq	24(%r14), %rdx
	movl	-308(%rbp), %eax
	leal	2(%r13), %r11d
	vmovsd	-128(%rbp), %xmm4
	leal	6(%r13), %r15d
	movl	%r11d, -192(%rbp)
	leal	3(%r13), %r11d
	vmovsd	-120(%rbp), %xmm14
	vxorps	%xmm2, %xmm2, %xmm2
	leal	(%r9,%rax), %r8d
	vmovsd	%xmm3, -208(%rbp)
	vmovsd	-104(%rbp), %xmm3
	vmovss	-156(%rbp), %xmm13
	leal	(%r8,%r12,2), %eax
	movl	%r11d, -180(%rbp)
	leal	4(%r13), %r11d
	vmovss	-152(%rbp), %xmm12
	movl	%eax, -200(%rbp)
	addl	%r12d, %eax
	vmovss	-148(%rbp), %xmm11
	leal	(%rax,%r12), %r10d
	vmovsd	%xmm3, -216(%rbp)
	vmovsd	-96(%rbp), %xmm3
	vmovss	-144(%rbp), %xmm10
	leal	(%r10,%r12), %esi
	movl	%eax, -260(%rbp)
	vmovss	-140(%rbp), %xmm9
	movl	%esi, -252(%rbp)
	movslq	-276(%rbp), %rsi
	vmovsd	%xmm3, -224(%rbp)
	vmovsd	-88(%rbp), %xmm3
	vmovss	-136(%rbp), %xmm8
	salq	$2, %rsi
	testl	%r13d, %r13d
	vmovss	-132(%rbp), %xmm7
	cmovs	%edi, %ecx
	vmovsd	%xmm3, -232(%rbp)
	vmovsd	-80(%rbp), %xmm3
	movl	%ecx, %eax
	vmovsd	%xmm3, -240(%rbp)
	vmovsd	-72(%rbp), %xmm3
	movq	16(%r14), %rcx
	sarl	$3, %eax
	imull	-272(%rbp), %eax
	vmovsd	%xmm3, -248(%rbp)
	sall	$3, %eax
	cltq
	leaq	(%rdx,%rax,4), %rax
	leal	1(%r13), %edx
	movl	%edx, -284(%rbp)
	movl	%r11d, -176(%rbp)
	movl	-252(%rbp), %edx
	leal	5(%r13), %r11d
	vcvtss2sd	-160(%rbp), %xmm1, %xmm3
	movl	%r11d, -168(%rbp)
	movl	%r8d, %r11d
	addl	%r12d, %edx
	subl	%edx, %r11d
	movl	%r11d, -184(%rbp)
	movl	%r8d, %r11d
	subl	-252(%rbp), %r11d
	movl	%r11d, -252(%rbp)
	movl	%r8d, %r11d
	subl	%r10d, %r11d
	movl	%r11d, -256(%rbp)
	movl	%r8d, %r11d
	subl	-260(%rbp), %r11d
	movl	%r11d, -260(%rbp)
	movl	-284(%rbp), %r11d
	.p2align 4,,10
	.p2align 3
.L47:
	vcvtss2sd	(%rcx,%rsi), %xmm1, %xmm0
	vsubsd	%xmm4, %xmm0, %xmm0
	vmulsd	%xmm3, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, (%rax)
	vmovaps	%xmm5, %xmm0
	cmpl	%r11d, %ebx
	jle	.L38
	movl	-184(%rbp), %r10d
	vcvtss2sd	%xmm13, %xmm13, %xmm15
	addl	%edx, %r10d
	movslq	%r10d, %r10
	vcvtss2sd	(%rcx,%r10,4), %xmm1, %xmm0
	vsubsd	%xmm14, %xmm0, %xmm0
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
.L38:
	vmovss	%xmm0, 4(%rax)
	vmovaps	%xmm2, %xmm0
	cmpl	-192(%rbp), %ebx
	jle	.L39
	movl	-252(%rbp), %r10d
	vcvtss2sd	%xmm12, %xmm12, %xmm15
	addl	%edx, %r10d
	movslq	%r10d, %r10
	vcvtss2sd	(%rcx,%r10,4), %xmm1, %xmm0
	vsubsd	-208(%rbp), %xmm0, %xmm0
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
.L39:
	vmovss	%xmm0, 8(%rax)
	vmovaps	%xmm2, %xmm0
	cmpl	-180(%rbp), %ebx
	jle	.L40
	movl	-256(%rbp), %r10d
	vcvtss2sd	%xmm11, %xmm11, %xmm15
	addl	%edx, %r10d
	movslq	%r10d, %r10
	vcvtss2sd	(%rcx,%r10,4), %xmm1, %xmm0
	vsubsd	-216(%rbp), %xmm0, %xmm0
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
.L40:
	vmovss	%xmm0, 12(%rax)
	vmovaps	%xmm2, %xmm0
	cmpl	-176(%rbp), %ebx
	jle	.L41
	movl	-260(%rbp), %r10d
	vcvtss2sd	%xmm10, %xmm10, %xmm15
	addl	%edx, %r10d
	movslq	%r10d, %r10
	vcvtss2sd	(%rcx,%r10,4), %xmm1, %xmm0
	vsubsd	-224(%rbp), %xmm0, %xmm0
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
.L41:
	vmovss	%xmm0, 16(%rax)
	vmovaps	%xmm2, %xmm0
	cmpl	-168(%rbp), %ebx
	jle	.L42
	movl	%r8d, %r10d
	subl	-200(%rbp), %r10d
	vcvtss2sd	%xmm9, %xmm9, %xmm15
	addl	%edx, %r10d
	movslq	%r10d, %r10
	vcvtss2sd	(%rcx,%r10,4), %xmm1, %xmm0
	vsubsd	-232(%rbp), %xmm0, %xmm0
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
.L42:
	vmovss	%xmm0, 20(%rax)
	vmovaps	%xmm2, %xmm0
	cmpl	%r15d, %ebx
	jle	.L43
	movl	%edx, %r10d
	vcvtss2sd	%xmm8, %xmm8, %xmm15
	subl	%r12d, %r10d
	movslq	%r10d, %r10
	vcvtss2sd	(%rcx,%r10,4), %xmm1, %xmm0
	vsubsd	-240(%rbp), %xmm0, %xmm0
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
.L43:
	vmovss	%xmm0, 24(%rax)
	cmpl	%edi, %ebx
	jle	.L44
	movslq	%edx, %r10
	vcvtss2sd	%xmm7, %xmm7, %xmm15
	addl	$1, %edx
	addq	$4, %rsi
	vcvtss2sd	(%rcx,%r10,4), %xmm1, %xmm0
	vsubsd	-248(%rbp), %xmm0, %xmm0
	addq	$32, %rax
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, -4(%rax)
	cmpl	%edx, %r9d
	jne	.L47
.L46:
	movl	-272(%rbp), %edi
	cmpl	%edi, -280(%rbp)
	jge	.L49
	leal	7(%r13), %eax
	leal	-1(%rdi), %edx
	vxorps	%xmm0, %xmm0, %xmm0
	testl	%r13d, %r13d
	cmovns	%r13d, %eax
	subl	-280(%rbp), %edx
	addq	$1, %rdx
	sarl	$3, %eax
	salq	$5, %rdx
	imull	%edi, %eax
	cltq
	addq	-304(%rbp), %rax
	salq	$5, %rax
	addq	24(%r14), %rax
	addq	%rax, %rdx
	.p2align 4,,10
	.p2align 3
.L48:
	vmovups	%ymm0, (%rax)
	addq	$32, %rax
	cmpq	%rdx, %rax
	jne	.L48
.L49:
	movl	-296(%rbp), %edi
	movl	-264(%rbp), %esi
	addl	%edi, -276(%rbp)
	movl	%esi, %r13d
	addl	%edi, %r9d
	cmpl	-292(%rbp), %esi
	jge	.L84
	leal	8(%rsi), %eax
	movl	%eax, -264(%rbp)
	jmp	.L14
	.p2align 4,,10
	.p2align 3
.L44:
	addl	$1, %edx
	movl	$0x00000000, 28(%rax)
	addq	$4, %rsi
	addq	$32, %rax
	cmpl	%r9d, %edx
	jne	.L47
	jmp	.L46
.L84:
	movl	%esi, %r11d
	movq	%r14, %r10
	movl	%ebx, %r9d
	movslq	%r12d, %r8
	cmpl	%esi, -288(%rbp)
	jg	.L11
.L88:
	vzeroupper
.L8:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L91
	addq	$272, %rsp
	popq	%rbx
	popq	%r10
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L11:
	.cfi_restore_state
	movl	%r11d, %r14d
	vxorpd	%xmm3, %xmm3, %xmm3
	vxorps	%xmm2, %xmm2, %xmm2
	movl	%r9d, %ebx
	leal	0(,%r8,8), %eax
	movl	%r8d, %r13d
	movl	%eax, -260(%rbp)
	leal	8(%r11), %eax
	movl	%eax, -264(%rbp)
	imull	%r8d, %eax
	movl	%eax, %r12d
	xorl	%eax, %eax
	testl	%r8d, %r8d
	cmovns	%r8, %rax
	movl	%r12d, %r11d
	movq	%r10, %r12
	movl	%eax, -256(%rbp)
	movq	%rax, -272(%rbp)
	.p2align 4,,10
	.p2align 3
.L36:
	movl	%r11d, %eax
	subl	-260(%rbp), %eax
	vpxor	%xmm0, %xmm0, %xmm0
	xorl	%ecx, %ecx
	movl	%eax, -180(%rbp)
	leal	-1(%r13), %r15d
	movl	%eax, %esi
	vmovdqa	%xmm0, -128(%rbp)
	addq	$2, %r15
	vmovdqa	%xmm0, -112(%rbp)
	vmovdqa	%xmm0, -96(%rbp)
	vmovdqa	%xmm0, -80(%rbp)
	vmovdqa	%xmm0, -160(%rbp)
	vmovdqa	%xmm0, -144(%rbp)
	.p2align 4,,10
	.p2align 3
.L21:
	leal	(%r14,%rcx), %eax
	cmpl	%ebx, %eax
	jge	.L16
	leaq	-160(%rbp), %rdi
	vmovss	(%rdi,%rcx,4), %xmm5
	testl	%r13d, %r13d
	jle	.L23
	movq	16(%r12), %rdx
	leaq	-128(%rbp), %r8
	movslq	%esi, %rax
	vmovsd	(%r8,%rcx,8), %xmm6
	leaq	(%rdx,%rax,4), %rdx
	movl	$1, %eax
	.p2align 4,,10
	.p2align 3
.L22:
	vcvtsi2sdl	%eax, %xmm1, %xmm4
	vcvtss2sd	-4(%rdx,%rax,4), %xmm1, %xmm0
	vsubsd	%xmm6, %xmm0, %xmm7
	vcvtss2sd	%xmm5, %xmm5, %xmm5
	addq	$1, %rax
	vdivsd	%xmm4, %xmm7, %xmm4
	vaddsd	%xmm4, %xmm6, %xmm6
	vsubsd	%xmm6, %xmm0, %xmm0
	vfmadd231sd	%xmm7, %xmm0, %xmm5
	vcvtsd2ss	%xmm5, %xmm5, %xmm5
	cmpq	%rax, %r15
	jne	.L22
	vmovsd	%xmm6, (%r8,%rcx,8)
.L23:
	vcvtss2sd	%xmm5, %xmm5, %xmm5
	vucomisd	%xmm5, %xmm3
	ja	.L85
	vsqrtsd	%xmm5, %xmm5, %xmm5
	vmovsd	.LC2(%rip), %xmm0
	addl	%r13d, %esi
	vdivsd	%xmm5, %xmm0, %xmm5
	vcvtsd2ss	%xmm5, %xmm5, %xmm5
	vmovss	%xmm5, (%rdi,%rcx,4)
	addq	$1, %rcx
	cmpq	$8, %rcx
	jne	.L21
.L16:
	movl	12(%r12), %edi
	movl	%edi, -184(%rbp)
	testl	%r13d, %r13d
	jle	.L24
	movl	-180(%rbp), %edx
	testl	%r14d, %r14d
	movl	%r14d, %r8d
	leal	4(%r14), %r15d
	vmovsd	-88(%rbp), %xmm6
	vmovsd	-80(%rbp), %xmm5
	movl	%r15d, -168(%rbp)
	leal	5(%r14), %r15d
	leal	(%rdx,%r13,2), %eax
	vmovsd	-72(%rbp), %xmm4
	movl	%r15d, -176(%rbp)
	leal	6(%r14), %r15d
	leal	0(%r13,%rax), %r9d
	movl	%eax, -248(%rbp)
	vmovsd	-120(%rbp), %xmm14
	leal	0(%r13,%r9), %r10d
	vmovsd	%xmm6, -208(%rbp)
	vmovss	-156(%rbp), %xmm13
	vmovsd	-112(%rbp), %xmm12
	leal	0(%r13,%r10), %esi
	vmovsd	%xmm5, -200(%rbp)
	vmovss	-152(%rbp), %xmm11
	vmovsd	-104(%rbp), %xmm10
	leal	0(%r13,%rsi), %ecx
	movl	%esi, -224(%rbp)
	vmovss	-148(%rbp), %xmm9
	movl	%ecx, -216(%rbp)
	leal	0(%r13,%rcx), %eax
	leal	7(%r14), %ecx
	movq	24(%r12), %rsi
	cmovs	%ecx, %r8d
	subl	-216(%rbp), %edx
	movl	%eax, -252(%rbp)
	movl	%edx, -216(%rbp)
	movl	-180(%rbp), %edx
	subl	-224(%rbp), %edx
	movl	%r8d, %eax
	vmovsd	%xmm4, -192(%rbp)
	vmovsd	-96(%rbp), %xmm8
	movl	%edx, -224(%rbp)
	sarl	$3, %eax
	movl	-180(%rbp), %edx
	leal	1(%r14), %r8d
	imull	%edi, %eax
	vmovss	-144(%rbp), %xmm7
	vmovss	-140(%rbp), %xmm6
	leal	2(%r14), %edi
	subl	%r10d, %edx
	vmovss	-136(%rbp), %xmm5
	vmovss	-132(%rbp), %xmm4
	movl	%edx, -232(%rbp)
	movl	-180(%rbp), %r10d
	sall	$3, %eax
	movl	%r10d, %edx
	subl	-248(%rbp), %r10d
	cltq
	subl	%r9d, %edx
	movl	%r10d, -248(%rbp)
	leaq	(%rsi,%rax,4), %rax
	leal	3(%r14), %esi
	movl	%edx, -240(%rbp)
	movl	-252(%rbp), %edx
	.p2align 4,,10
	.p2align 3
.L33:
	movl	$0x00000000, (%rax)
	vmovaps	%xmm2, %xmm0
	cmpl	%ebx, %r8d
	jge	.L25
	movl	-216(%rbp), %r9d
	movq	16(%r12), %r10
	vcvtss2sd	%xmm13, %xmm13, %xmm15
	addl	%edx, %r9d
	movslq	%r9d, %r9
	vcvtss2sd	(%r10,%r9,4), %xmm1, %xmm0
	vsubsd	%xmm14, %xmm0, %xmm0
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
.L25:
	vmovss	%xmm0, 4(%rax)
	vmovaps	%xmm2, %xmm0
	cmpl	%ebx, %edi
	jge	.L26
	movl	-224(%rbp), %r9d
	movq	16(%r12), %r10
	vcvtss2sd	%xmm11, %xmm11, %xmm15
	addl	%edx, %r9d
	movslq	%r9d, %r9
	vcvtss2sd	(%r10,%r9,4), %xmm1, %xmm0
	vsubsd	%xmm12, %xmm0, %xmm0
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
.L26:
	vmovss	%xmm0, 8(%rax)
	vmovaps	%xmm2, %xmm0
	cmpl	%ebx, %esi
	jge	.L27
	movl	-232(%rbp), %r9d
	movq	16(%r12), %r10
	vcvtss2sd	%xmm9, %xmm9, %xmm15
	addl	%edx, %r9d
	movslq	%r9d, %r9
	vcvtss2sd	(%r10,%r9,4), %xmm1, %xmm0
	vsubsd	%xmm10, %xmm0, %xmm0
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
.L27:
	vmovss	%xmm0, 12(%rax)
	vmovaps	%xmm2, %xmm0
	cmpl	%ebx, -168(%rbp)
	jge	.L28
	movl	-240(%rbp), %r9d
	movq	16(%r12), %r10
	vcvtss2sd	%xmm7, %xmm7, %xmm15
	addl	%edx, %r9d
	movslq	%r9d, %r9
	vcvtss2sd	(%r10,%r9,4), %xmm1, %xmm0
	vsubsd	%xmm8, %xmm0, %xmm0
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
.L28:
	vmovss	%xmm0, 16(%rax)
	vmovaps	%xmm2, %xmm0
	cmpl	-176(%rbp), %ebx
	jle	.L29
	movl	-248(%rbp), %r9d
	movq	16(%r12), %r10
	vcvtss2sd	%xmm6, %xmm6, %xmm15
	addl	%edx, %r9d
	movslq	%r9d, %r9
	vcvtss2sd	(%r10,%r9,4), %xmm1, %xmm0
	vsubsd	-208(%rbp), %xmm0, %xmm0
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
.L29:
	vmovss	%xmm0, 20(%rax)
	vmovaps	%xmm2, %xmm0
	cmpl	%ebx, %r15d
	jge	.L30
	movl	%edx, %r9d
	movq	16(%r12), %r10
	vcvtss2sd	%xmm5, %xmm5, %xmm15
	subl	%r13d, %r9d
	movslq	%r9d, %r9
	vcvtss2sd	(%r10,%r9,4), %xmm1, %xmm0
	vsubsd	-200(%rbp), %xmm0, %xmm0
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
.L30:
	vmovss	%xmm0, 24(%rax)
	cmpl	%ebx, %ecx
	jge	.L31
	movq	16(%r12), %r9
	movslq	%edx, %r10
	vcvtss2sd	%xmm4, %xmm4, %xmm15
	addl	$1, %edx
	addq	$32, %rax
	vcvtss2sd	(%r9,%r10,4), %xmm1, %xmm0
	vsubsd	-192(%rbp), %xmm0, %xmm0
	vmulsd	%xmm15, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, -4(%rax)
	cmpl	%edx, %r11d
	jne	.L33
.L24:
	movl	-256(%rbp), %edi
	movl	-184(%rbp), %esi
	cmpl	%esi, %edi
	jge	.L34
	leal	7(%r14), %eax
	leal	-1(%rsi), %edx
	vxorps	%xmm0, %xmm0, %xmm0
	testl	%r14d, %r14d
	cmovns	%r14d, %eax
	subl	%edi, %edx
	addq	$1, %rdx
	sarl	$3, %eax
	salq	$5, %rdx
	imull	%esi, %eax
	cltq
	addq	-272(%rbp), %rax
	salq	$5, %rax
	addq	24(%r12), %rax
	addq	%rax, %rdx
	.p2align 4,,10
	.p2align 3
.L35:
	vmovups	%ymm0, (%rax)
	addq	$32, %rax
	cmpq	%rax, %rdx
	jne	.L35
.L34:
	movl	-264(%rbp), %edi
	movl	-180(%rbp), %r11d
	movl	%r13d, %eax
	sall	$4, %eax
	movl	%edi, %r14d
	addl	%eax, %r11d
	cmpl	%edi, -288(%rbp)
	jle	.L88
	leal	8(%rdi), %eax
	movl	%eax, -264(%rbp)
	jmp	.L36
	.p2align 4,,10
	.p2align 3
.L31:
	addl	$1, %edx
	movl	$0x00000000, 28(%rax)
	addq	$32, %rax
	cmpl	%r11d, %edx
	jne	.L33
	jmp	.L24
.L9:
	addl	$1, %eax
	xorl	%edx, %edx
	jmp	.L59
.L85:
	vmovsd	%xmm5, %xmm5, %xmm0
	movl	%r11d, -200(%rbp)
	movl	%esi, -192(%rbp)
	movq	%rcx, -176(%rbp)
	movq	%rdi, -168(%rbp)
	vzeroupper
	call	sqrt@PLT
	movq	-176(%rbp), %rcx
	vxorpd	%xmm3, %xmm3, %xmm3
	vmovsd	.LC2(%rip), %xmm4
	movq	-168(%rbp), %rdi
	movl	-192(%rbp), %esi
	vxorps	%xmm2, %xmm2, %xmm2
	vxorps	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm0, %xmm4, %xmm0
	movl	-200(%rbp), %r11d
	addl	%r13d, %esi
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, (%rdi,%rcx,4)
	addq	$1, %rcx
	cmpq	$8, %rcx
	jne	.L21
	jmp	.L16
.L91:
	call	__stack_chk_fail@PLT
	.p2align 4,,10
	.p2align 3
.L86:
	vmovsd	%xmm4, %xmm4, %xmm0
	movq	%rdi, -192(%rbp)
	movl	%r9d, -180(%rbp)
	movl	%esi, -176(%rbp)
	movq	%rcx, -168(%rbp)
	vzeroupper
	call	sqrt@PLT
	movq	-168(%rbp), %rcx
	vxorps	%xmm5, %xmm5, %xmm5
	vmovsd	.LC2(%rip), %xmm6
	movq	-192(%rbp), %rdi
	movl	-176(%rbp), %esi
	vxorps	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm0, %xmm6, %xmm0
	movl	-180(%rbp), %r9d
	vxorpd	%xmm6, %xmm6, %xmm6
	addl	%r12d, %esi
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, (%rdi,%rcx,4)
	addq	$1, %rcx
	cmpq	$8, %rcx
	jne	.L12
	jmp	.L56
	.cfi_endproc
.LFE8709:
	.size	_ZN6Matrix24standardize_rows_swizzleEv._omp_fn.0, .-_ZN6Matrix24standardize_rows_swizzleEv._omp_fn.0
	.p2align 4
	.type	_ZNK6Matrix16covariance_tiledEv._omp_fn.0, @function
_ZNK6Matrix16covariance_tiledEv._omp_fn.0:
.LFB8710:
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
	subq	$256, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	(%rdi), %r14
	movq	%fs:40, %rax
	movq	%rax, 248(%rsp)
	xorl	%eax, %eax
	movl	8(%r14), %eax
	movl	%eax, 60(%rsp)
	testl	%eax, %eax
	jle	.L137
	movl	%eax, %ebx
	leal	127(%rax), %eax
	leal	255(%rbx), %esi
	sarl	$7, %eax
	sarl	$8, %esi
	cltq
	movslq	%esi, %rsi
	movq	%rax, 48(%rsp)
	imulq	%rax, %rsi
.L118:
	leaq	232(%rsp), %rax
	leaq	240(%rsp), %r9
	xorl	%edi, %edi
	movl	$1, %ecx
	movq	%rax, %r8
	movl	$1, %edx
	movq	%r9, 24(%rsp)
	movq	%rax, 32(%rsp)
	call	GOMP_loop_nonmonotonic_dynamic_start@PLT
	testb	%al, %al
	je	.L98
	movq	%r14, %r13
.L94:
	movq	232(%rsp), %rax
	movq	240(%rsp), %rbx
	cqto
	movq	%rax, 64(%rsp)
	idivq	48(%rsp)
	movq	%rbx, 40(%rsp)
	movq	%r13, %rbx
	sall	$8, %eax
	movl	%edx, %r15d
	movl	%eax, 104(%rsp)
	sall	$7, %r15d
	movl	%r15d, %r12d
.L95:
	cmpl	104(%rsp), %r12d
	jge	.L96
.L100:
	addq	$1, 64(%rsp)
	movq	64(%rsp), %rax
	cmpq	%rax, 40(%rsp)
	jle	.L138
	subl	$-128, %r12d
	cmpl	%r12d, 60(%rsp)
	jg	.L95
	addl	$256, 104(%rsp)
	xorl	%r12d, %r12d
	cmpl	104(%rsp), %r12d
	jl	.L100
.L96:
	movl	12(%rbx), %r15d
	testl	%r15d, %r15d
	jle	.L100
	movl	$0, 72(%rsp)
	movslq	%r12d, %r13
	movl	%r12d, %r10d
	movq	%rbx, %r11
	movq	%r13, %r12
	movl	%r15d, %ebx
.L107:
	movl	$224, %edx
	movl	%ebx, %eax
	subl	72(%rsp), %eax
	cmpl	%edx, %eax
	cmovg	%edx, %eax
	movl	8(%r11), %edx
	movl	%edx, %edi
	subl	%r10d, %edx
	subl	104(%rsp), %edi
	movl	%edi, 76(%rsp)
	movl	%edx, 108(%rsp)
	cmpl	$255, %edi
	jle	.L139
	cmpl	$127, 108(%rsp)
	jle	.L136
	movl	$128, 108(%rsp)
.L136:
	movl	$256, 76(%rsp)
.L119:
	movl	72(%rsp), %edi
	subl	$1, %eax
	movl	$-16, 80(%rsp)
	salq	$3, %rax
	leal	0(,%rdi,8), %edx
	movq	%rax, 112(%rsp)
	movl	104(%rsp), %edi
	movslq	%edx, %r14
	leal	1(%rdi), %r13d
.L108:
	movl	108(%rsp), %eax
	testl	%eax, %eax
	jle	.L106
	movl	%r13d, %eax
	leal	6(%r13), %r15d
	movl	80(%rsp), %r9d
	subl	$1, %eax
	movl	%eax, 124(%rsp)
	cmovs	%r15d, %eax
	sarl	$3, %eax
	movl	%eax, 120(%rsp)
	testl	%r9d, %r9d
	jle	.L122
	leal	1(%r13), %edi
	movl	108(%rsp), %eax
	movl	%edi, 100(%rsp)
	leal	2(%r13), %edi
	movl	%edi, 96(%rsp)
	leal	3(%r13), %edi
	cmpl	%r9d, %eax
	movl	%edi, 92(%rsp)
	leal	4(%r13), %edi
	cmovle	%eax, %r9d
	xorl	%eax, %eax
	movl	%edi, 88(%rsp)
	leal	5(%r13), %edi
	movl	%edi, 84(%rsp)
	jmp	.L111
	.p2align 4,,10
	.p2align 3
.L141:
	leal	16(%rax), %edi
	addq	$16, %rax
	cmpl	%eax, %r9d
	jle	.L140
.L111:
	cmpl	104(%rsp), %r10d
	je	.L141
	movl	120(%rsp), %esi
	leal	(%r10,%rax), %r8d
	movq	24(%r11), %rdi
	vxorps	%xmm12, %xmm12, %xmm12
	leal	7(%r8), %edx
	leal	0(,%rbx,8), %ecx
	vmovaps	%ymm12, %ymm13
	vmovaps	%ymm12, 128(%rsp)
	imull	%ebx, %esi
	vmovaps	%ymm12, %ymm14
	vmovaps	%ymm12, %ymm10
	movslq	%ecx, %rcx
	vmovaps	%ymm12, %ymm11
	vmovaps	%ymm12, %ymm15
	vmovaps	%ymm12, %ymm7
	vmovaps	%ymm12, 192(%rsp)
	vmovaps	%ymm12, %ymm4
	vmovaps	%ymm12, %ymm3
	vmovaps	%ymm12, %ymm5
	vmovaps	%ymm12, 160(%rsp)
	vmovaps	%ymm12, %ymm6
	sall	$3, %esi
	vmovaps	%ymm12, %ymm8
	vmovaps	%ymm12, %ymm9
	movslq	%esi, %rsi
	addq	%r14, %rsi
	testl	%r8d, %r8d
	cmovns	%r8d, %edx
	sarl	$3, %edx
	imull	%ebx, %edx
	sall	$3, %edx
	movslq	%edx, %rdx
	addq	%r14, %rdx
	leaq	(%rdi,%rdx,4), %r8
	leaq	(%rdi,%rsi,4), %rdx
	addq	112(%rsp), %rsi
	leaq	32(%rdi,%rsi,4), %rsi
	.p2align 4,,10
	.p2align 3
.L115:
	vmovaps	(%r8), %ymm1
	vbroadcastss	(%rdx), %ymm2
	addq	$32, %rdx
	vmovaps	(%r8,%rcx,4), %ymm0
	addq	$32, %r8
	vfmadd231ps	%ymm2, %ymm1, %ymm4
	vfmadd213ps	192(%rsp), %ymm0, %ymm2
	vmovaps	%ymm2, 192(%rsp)
	vbroadcastss	-28(%rdx), %ymm2
	vfmadd231ps	%ymm2, %ymm1, %ymm9
	vfmadd213ps	128(%rsp), %ymm0, %ymm2
	vmovaps	%ymm2, 128(%rsp)
	vbroadcastss	-24(%rdx), %ymm2
	vfmadd231ps	%ymm2, %ymm1, %ymm8
	vfmadd231ps	%ymm2, %ymm0, %ymm15
	vbroadcastss	-20(%rdx), %ymm2
	vfmadd231ps	%ymm2, %ymm1, %ymm7
	vfmadd231ps	%ymm2, %ymm0, %ymm11
	vbroadcastss	-16(%rdx), %ymm2
	vfmadd231ps	%ymm2, %ymm1, %ymm6
	vfmadd231ps	%ymm2, %ymm0, %ymm10
	vbroadcastss	-12(%rdx), %ymm2
	vfmadd231ps	%ymm2, %ymm1, %ymm5
	vfmadd231ps	%ymm2, %ymm0, %ymm14
	vbroadcastss	-8(%rdx), %ymm2
	vfmadd231ps	%ymm2, %ymm1, %ymm3
	vfmadd231ps	%ymm2, %ymm0, %ymm13
	vbroadcastss	-4(%rdx), %ymm2
	vfmadd213ps	160(%rsp), %ymm2, %ymm1
	vfmadd231ps	%ymm2, %ymm0, %ymm12
	vmovaps	%ymm1, 160(%rsp)
	cmpq	%rsi, %rdx
	jne	.L115
	movl	124(%rsp), %edx
	imull	8(%r11), %edx
	leal	16(%rax), %edi
	movq	32(%r11), %rcx
	movslq	%edx, %rdx
	addq	%r12, %rdx
	addq	%rax, %rdx
	leaq	(%rcx,%rdx,4), %rdx
	vaddps	(%rdx), %ymm4, %ymm0
	vmovaps	192(%rsp), %ymm4
	vmovaps	%ymm0, (%rdx)
	vaddps	32(%rdx), %ymm4, %ymm0
	vmovaps	128(%rsp), %ymm4
	vmovaps	%ymm0, 32(%rdx)
	movl	8(%r11), %edx
	movq	32(%r11), %rcx
	imull	%r13d, %edx
	movslq	%edx, %rdx
	addq	%r12, %rdx
	addq	%rax, %rdx
	leaq	(%rcx,%rdx,4), %rdx
	vaddps	32(%rdx), %ymm4, %ymm0
	vaddps	(%rdx), %ymm9, %ymm9
	vmovaps	%ymm0, 32(%rdx)
	vmovaps	%ymm9, (%rdx)
	movl	100(%rsp), %edx
	imull	8(%r11), %edx
	movq	32(%r11), %rcx
	movslq	%edx, %rdx
	addq	%r12, %rdx
	addq	%rax, %rdx
	leaq	(%rcx,%rdx,4), %rdx
	vaddps	(%rdx), %ymm8, %ymm8
	vaddps	32(%rdx), %ymm15, %ymm15
	vmovaps	%ymm8, (%rdx)
	vmovaps	%ymm15, 32(%rdx)
	movl	96(%rsp), %edx
	imull	8(%r11), %edx
	movq	32(%r11), %rcx
	movslq	%edx, %rdx
	addq	%r12, %rdx
	addq	%rax, %rdx
	leaq	(%rcx,%rdx,4), %rdx
	vaddps	(%rdx), %ymm7, %ymm7
	vaddps	32(%rdx), %ymm11, %ymm11
	vmovaps	%ymm7, (%rdx)
	vmovaps	%ymm11, 32(%rdx)
	movl	92(%rsp), %edx
	imull	8(%r11), %edx
	movq	32(%r11), %rcx
	movslq	%edx, %rdx
	addq	%r12, %rdx
	addq	%rax, %rdx
	leaq	(%rcx,%rdx,4), %rdx
	vaddps	(%rdx), %ymm6, %ymm6
	vaddps	32(%rdx), %ymm10, %ymm10
	vmovaps	%ymm6, (%rdx)
	vmovaps	%ymm10, 32(%rdx)
	movl	88(%rsp), %edx
	imull	8(%r11), %edx
	movq	32(%r11), %rcx
	movslq	%edx, %rdx
	addq	%r12, %rdx
	addq	%rax, %rdx
	leaq	(%rcx,%rdx,4), %rdx
	vaddps	(%rdx), %ymm5, %ymm5
	vaddps	32(%rdx), %ymm14, %ymm14
	vmovaps	%ymm5, (%rdx)
	vmovaps	160(%rsp), %ymm5
	vmovaps	%ymm14, 32(%rdx)
	movl	84(%rsp), %edx
	imull	8(%r11), %edx
	movq	32(%r11), %rcx
	movslq	%edx, %rdx
	addq	%r12, %rdx
	addq	%rax, %rdx
	leaq	(%rcx,%rdx,4), %rdx
	vaddps	(%rdx), %ymm3, %ymm3
	vaddps	32(%rdx), %ymm13, %ymm13
	vmovaps	%ymm3, (%rdx)
	vmovaps	%ymm13, 32(%rdx)
	movl	8(%r11), %edx
	movq	32(%r11), %rcx
	imull	%r15d, %edx
	movslq	%edx, %rdx
	addq	%r12, %rdx
	addq	%rax, %rdx
	addq	$16, %rax
	leaq	(%rcx,%rdx,4), %rdx
	vaddps	(%rdx), %ymm5, %ymm4
	vaddps	32(%rdx), %ymm12, %ymm12
	vmovaps	%ymm4, (%rdx)
	vmovaps	%ymm12, 32(%rdx)
	movl	12(%r11), %ebx
	cmpl	%eax, %r9d
	jg	.L111
.L140:
	cmpl	%edi, 108(%rsp)
	jg	.L109
.L106:
	movl	80(%rsp), %eax
	addl	$8, %r13d
	leal	8(%rax), %edx
	addl	$24, %eax
	cmpl	%eax, 76(%rsp)
	jle	.L104
	movl	%edx, 80(%rsp)
	jmp	.L108
.L138:
	movq	24(%rsp), %rsi
	movq	32(%rsp), %rdi
	movq	%rbx, %r13
	vzeroupper
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	testb	%al, %al
	jne	.L94
.L98:
	call	GOMP_loop_end_nowait@PLT
	movq	248(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L142
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
.L122:
	.cfi_restore_state
	xorl	%edi, %edi
.L109:
	leal	1(%r13), %eax
	movslq	%edi, %rcx
	movl	%eax, 100(%rsp)
	leal	2(%r13), %eax
	movl	%eax, 96(%rsp)
	leal	3(%r13), %eax
	movl	%eax, 92(%rsp)
	leal	4(%r13), %eax
	movl	%eax, 88(%rsp)
	leal	5(%r13), %eax
	movl	%eax, 84(%rsp)
	.p2align 4,,10
	.p2align 3
.L113:
	movl	120(%rsp), %edx
	leal	(%r10,%rdi), %r9d
	movq	24(%r11), %r8
	vxorps	%xmm3, %xmm3, %xmm3
	leal	7(%r9), %eax
	leal	0(,%rbx,8), %esi
	vmovaps	%ymm3, %ymm5
	vmovaps	%ymm3, 192(%rsp)
	imull	%ebx, %edx
	vmovaps	%ymm3, %ymm7
	vmovaps	%ymm3, %ymm9
	movslq	%esi, %rsi
	vmovaps	%ymm3, %ymm11
	vmovaps	%ymm3, %ymm13
	vmovaps	%ymm3, %ymm4
	vmovaps	%ymm3, 160(%rsp)
	vmovaps	%ymm3, %ymm6
	vmovaps	%ymm3, %ymm8
	vmovaps	%ymm3, %ymm10
	vmovaps	%ymm3, 128(%rsp)
	vmovaps	%ymm3, %ymm12
	sall	$3, %edx
	vmovaps	%ymm3, %ymm14
	vmovaps	%ymm3, %ymm15
	movslq	%edx, %rdx
	addq	%r14, %rdx
	testl	%r9d, %r9d
	cmovns	%r9d, %eax
	sarl	$3, %eax
	imull	%ebx, %eax
	sall	$3, %eax
	cltq
	addq	%r14, %rax
	leaq	(%r8,%rax,4), %r9
	leaq	(%r8,%rdx,4), %rax
	addq	112(%rsp), %rdx
	leaq	32(%r8,%rdx,4), %rdx
	.p2align 4,,10
	.p2align 3
.L112:
	vmovaps	(%r9), %ymm1
	vbroadcastss	(%rax), %ymm2
	addq	$32, %rax
	vmovaps	(%r9,%rsi,4), %ymm0
	addq	$32, %r9
	vfmadd231ps	%ymm2, %ymm1, %ymm3
	vfmadd213ps	128(%rsp), %ymm0, %ymm2
	vmovaps	%ymm2, 128(%rsp)
	vbroadcastss	-28(%rax), %ymm2
	vfmadd231ps	%ymm2, %ymm1, %ymm15
	vfmadd213ps	160(%rsp), %ymm0, %ymm2
	vmovaps	%ymm2, 160(%rsp)
	vbroadcastss	-24(%rax), %ymm2
	vfmadd231ps	%ymm2, %ymm0, %ymm13
	vfmadd231ps	%ymm2, %ymm1, %ymm14
	vbroadcastss	-20(%rax), %ymm2
	vfmadd231ps	%ymm2, %ymm0, %ymm11
	vfmadd231ps	%ymm2, %ymm1, %ymm12
	vbroadcastss	-16(%rax), %ymm2
	vfmadd231ps	%ymm2, %ymm0, %ymm9
	vfmadd231ps	%ymm2, %ymm1, %ymm10
	vbroadcastss	-12(%rax), %ymm2
	vfmadd231ps	%ymm2, %ymm0, %ymm7
	vfmadd231ps	%ymm2, %ymm1, %ymm8
	vbroadcastss	-8(%rax), %ymm2
	vfmadd231ps	%ymm2, %ymm0, %ymm5
	vfmadd231ps	%ymm2, %ymm1, %ymm6
	vbroadcastss	-4(%rax), %ymm2
	vfmadd213ps	192(%rsp), %ymm2, %ymm0
	vfmadd231ps	%ymm2, %ymm1, %ymm4
	vmovaps	%ymm0, 192(%rsp)
	cmpq	%rdx, %rax
	jne	.L112
	movl	124(%rsp), %eax
	imull	8(%r11), %eax
	addl	$16, %edi
	movq	32(%r11), %rdx
	cltq
	addq	%r12, %rax
	addq	%rcx, %rax
	leaq	(%rdx,%rax,4), %rax
	vaddps	(%rax), %ymm3, %ymm0
	vmovaps	128(%rsp), %ymm3
	vmovaps	%ymm0, (%rax)
	vaddps	32(%rax), %ymm3, %ymm0
	vmovaps	160(%rsp), %ymm3
	vmovaps	%ymm0, 32(%rax)
	movl	8(%r11), %eax
	movq	32(%r11), %rdx
	imull	%r13d, %eax
	cltq
	addq	%r12, %rax
	addq	%rcx, %rax
	leaq	(%rdx,%rax,4), %rax
	vaddps	32(%rax), %ymm3, %ymm0
	vaddps	(%rax), %ymm15, %ymm15
	vmovaps	%ymm0, 32(%rax)
	vmovaps	%ymm15, (%rax)
	movl	100(%rsp), %eax
	imull	8(%r11), %eax
	movq	32(%r11), %rdx
	cltq
	addq	%r12, %rax
	addq	%rcx, %rax
	leaq	(%rdx,%rax,4), %rax
	vaddps	(%rax), %ymm14, %ymm14
	vaddps	32(%rax), %ymm13, %ymm13
	vmovaps	%ymm14, (%rax)
	vmovaps	%ymm13, 32(%rax)
	movl	96(%rsp), %eax
	imull	8(%r11), %eax
	movq	32(%r11), %rdx
	cltq
	addq	%r12, %rax
	addq	%rcx, %rax
	leaq	(%rdx,%rax,4), %rax
	vaddps	(%rax), %ymm12, %ymm12
	vaddps	32(%rax), %ymm11, %ymm11
	vmovaps	%ymm12, (%rax)
	vmovaps	%ymm11, 32(%rax)
	movl	92(%rsp), %eax
	imull	8(%r11), %eax
	movq	32(%r11), %rdx
	cltq
	addq	%r12, %rax
	addq	%rcx, %rax
	leaq	(%rdx,%rax,4), %rax
	vaddps	(%rax), %ymm10, %ymm10
	vaddps	32(%rax), %ymm9, %ymm9
	vmovaps	%ymm10, (%rax)
	vmovaps	%ymm9, 32(%rax)
	movl	88(%rsp), %eax
	imull	8(%r11), %eax
	movq	32(%r11), %rdx
	cltq
	addq	%r12, %rax
	addq	%rcx, %rax
	leaq	(%rdx,%rax,4), %rax
	vaddps	(%rax), %ymm8, %ymm8
	vaddps	32(%rax), %ymm7, %ymm7
	vmovaps	%ymm8, (%rax)
	vmovaps	%ymm7, 32(%rax)
	movl	84(%rsp), %eax
	imull	8(%r11), %eax
	movq	32(%r11), %rdx
	cltq
	addq	%r12, %rax
	addq	%rcx, %rax
	leaq	(%rdx,%rax,4), %rax
	vaddps	32(%rax), %ymm5, %ymm5
	vaddps	(%rax), %ymm6, %ymm6
	vmovaps	%ymm5, 32(%rax)
	vmovaps	192(%rsp), %ymm5
	vmovaps	%ymm6, (%rax)
	movl	8(%r11), %eax
	movq	32(%r11), %rdx
	imull	%r15d, %eax
	cltq
	addq	%r12, %rax
	addq	%rcx, %rax
	addq	$16, %rcx
	leaq	(%rdx,%rax,4), %rax
	vaddps	(%rax), %ymm4, %ymm4
	vaddps	32(%rax), %ymm5, %ymm3
	vmovaps	%ymm4, (%rax)
	vmovaps	%ymm3, 32(%rax)
	movl	12(%r11), %ebx
	cmpl	%edi, 108(%rsp)
	jg	.L113
	jmp	.L106
.L139:
	movl	%edx, %edi
	movl	$128, %edx
	cmpl	%edx, %edi
	cmovle	%edi, %edx
	movl	%edx, 108(%rsp)
	movl	76(%rsp), %edx
	testl	%edx, %edx
	jg	.L119
.L104:
	addl	$224, 72(%rsp)
	movl	72(%rsp), %eax
	cmpl	%ebx, %eax
	jl	.L107
	movl	%r10d, %r12d
	movq	%r11, %rbx
	jmp	.L100
.L137:
	xorl	%esi, %esi
	jmp	.L118
.L142:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE8710:
	.size	_ZNK6Matrix16covariance_tiledEv._omp_fn.0, .-_ZNK6Matrix16covariance_tiledEv._omp_fn.0
	.p2align 4
	.type	_ZN6Matrix11storeResultEPf._omp_fn.0, @function
_ZN6Matrix11storeResultEPf._omp_fn.0:
.LFB8711:
	.cfi_startproc
	endbr64
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	subq	$8, %rsp
	.cfi_def_cfa_offset 48
	movq	8(%rdi), %rbx
	movl	(%rbx), %ebp
	testl	%ebp, %ebp
	jle	.L159
	movq	%rdi, %r13
	call	omp_get_num_threads@PLT
	movl	%eax, %r12d
	call	omp_get_thread_num@PLT
	movl	%ebp, %esi
	xorl	%edx, %edx
	imull	%ebp, %esi
	movl	%eax, %ecx
	movl	%esi, %eax
	divl	%r12d
	movl	%eax, %esi
	cmpl	%edx, %ecx
	jb	.L145
.L157:
	imull	%esi, %ecx
	leal	(%rcx,%rdx), %eax
	leal	(%rsi,%rax), %edx
	cmpl	%edx, %eax
	jnb	.L159
	xorl	%edx, %edx
	movq	0(%r13), %r8
	divl	%ebp
	cmpl	$1, %ebp
	jne	.L161
	subl	$1, %esi
	xorl	%edi, %edi
	jmp	.L152
	.p2align 4,,10
	.p2align 3
.L162:
	addl	$1, %eax
	addl	$1, %edi
	xorl	%edx, %edx
.L152:
	cmpl	%edx, %eax
	jg	.L150
	movl	8(%rbx), %ecx
	movq	32(%rbx), %r9
	imull	%eax, %ecx
	addl	%edx, %ecx
	addl	%eax, %edx
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vmovss	(%r9,%rcx,4), %xmm0
	vmovss	%xmm0, (%r8,%rdx,4)
.L150:
	cmpl	%esi, %edi
	jne	.L162
.L159:
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L145:
	.cfi_restore_state
	addl	$1, %esi
	xorl	%edx, %edx
	jmp	.L157
	.p2align 4,,10
	.p2align 3
.L161:
	subl	$1, %esi
	xorl	%edi, %edi
	jmp	.L149
	.p2align 4,,10
	.p2align 3
.L155:
	addl	$1, %edi
.L149:
	cmpl	%eax, %edx
	jl	.L156
	movl	8(%rbx), %ecx
	movq	32(%rbx), %r9
	imull	%eax, %ecx
	addl	%edx, %ecx
	movslq	%ecx, %rcx
	vmovss	(%r9,%rcx,4), %xmm0
	movl	%eax, %ecx
	imull	%ebp, %ecx
	addl	%edx, %ecx
	movslq	%ecx, %rcx
	vmovss	%xmm0, (%r8,%rcx,4)
.L156:
	cmpl	%esi, %edi
	je	.L159
	addl	$1, %edx
	cmpl	%edx, %ebp
	jg	.L155
	addl	$1, %eax
	xorl	%edx, %edx
	jmp	.L155
	.cfi_endproc
.LFE8711:
	.size	_ZN6Matrix11storeResultEPf._omp_fn.0, .-_ZN6Matrix11storeResultEPf._omp_fn.0
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC5:
	.string	"Wall time            : "
.LC6:
	.string	" s\n"
.LC7:
	.string	"Total FLOPs          : "
.LC8:
	.string	" GFLOP\n"
.LC9:
	.string	"Achieved FLOPS       : "
.LC10:
	.string	" GFLOP/s\n"
.LC11:
	.string	"Cycles elapsed       : "
.LC12:
	.string	"\n"
.LC13:
	.string	"Measured CPU freq    : "
.LC14:
	.string	" GHz\n"
.LC15:
	.string	"Checksum             : "
	.section	.text.unlikely,"ax",@progbits
.LCOLDB16:
	.text
.LHOTB16:
	.p2align 4
	.globl	_Z9correlateiiPKfPf
	.type	_Z9correlateiiPKfPf, @function
_Z9correlateiiPKfPf:
.LFB8024:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA8024
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	vmovd	%edi, %xmm6
	movl	%esi, %r15d
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	vpinsrd	$1, %esi, %xmm6, %xmm0
	movq	%rcx, %r14
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movl	%edi, %ebp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	leal	30(%rdi), %ebx
	subq	$120, %rsp
	.cfi_def_cfa_offset 176
	movq	%fs:40, %rax
	movq	%rax, 104(%rsp)
	xorl	%eax, %eax
	movl	%edi, %eax
	movl	$32, %edi
	movq	%rdx, 80(%rsp)
	addl	$15, %eax
	leaq	48(%rsp), %r13
	cmovns	%eax, %ebx
	andl	$-16, %ebx
	vmovd	%ebx, %xmm7
	vpinsrd	$1, %esi, %xmm7, %xmm1
	imull	%ebx, %esi
	imull	%ebx, %ebx
	vpunpcklqdq	%xmm1, %xmm0, %xmm0
	vmovdqa	%xmm0, 64(%rsp)
	movslq	%esi, %rsi
	salq	$2, %rsi
	call	aligned_alloc@PLT
	movslq	%ebx, %rsi
	movl	$32, %edi
	leaq	64(%rsp), %rbx
	salq	$2, %rsi
	movq	%rax, 88(%rsp)
	call	aligned_alloc@PLT
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	movq	%r13, %rsi
	leaq	_ZN6MatrixC2EiiPKf._omp_fn.0(%rip), %rdi
	movq	%rax, 96(%rsp)
	movq	%rbx, 48(%rsp)
	call	GOMP_parallel@PLT
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	movq	%r13, %rsi
	leaq	_ZN6Matrix24standardize_rows_swizzleEv._omp_fn.0(%rip), %rdi
	movq	%rbx, 48(%rsp)
	call	GOMP_parallel@PLT
	rdtscp
	salq	$32, %rdx
	movq	%rax, %r8
	movl	%ecx, 44(%rsp)
	orq	%rdx, %r8
	movq	%r8, 24(%rsp)
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	%r13, %rsi
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	leaq	_ZNK6Matrix16covariance_tiledEv._omp_fn.0(%rip), %rdi
	movq	%rax, 8(%rsp)
	movq	%rbx, 48(%rsp)
	call	GOMP_parallel@PLT
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	%rax, %rsi
	rdtscp
	vxorpd	%xmm3, %xmm3, %xmm3
	vmovsd	.LC3(%rip), %xmm2
	movq	8(%rsp), %r9
	movq	24(%rsp), %r8
	subq	%r9, %rsi
	vcvtsi2sdl	%r15d, %xmm3, %xmm1
	movq	%rax, %r12
	salq	$32, %rdx
	vcvtsi2sdq	%rsi, %xmm3, %xmm0
	orq	%rdx, %r12
	movl	%ecx, 44(%rsp)
	vdivsd	%xmm2, %xmm0, %xmm0
	vaddsd	.LC1(%rip), %xmm0, %xmm4
	vcvtsi2sdl	%ebp, %xmm3, %xmm0
	vmovsd	%xmm4, 8(%rsp)
	vmulsd	%xmm0, %xmm0, %xmm0
	vmulsd	%xmm1, %xmm0, %xmm0
	vmulsd	.LC4(%rip), %xmm0, %xmm5
	vdivsd	%xmm4, %xmm5, %xmm7
	vmovq	%xmm5, %r15
	vmovsd	%xmm7, 16(%rsp)
	subq	%r8, %r12
	js	.L164
	vcvtsi2sdq	%r12, %xmm3, %xmm0
.L165:
	vdivsd	8(%rsp), %xmm0, %xmm0
	vdivsd	%xmm2, %xmm0, %xmm2
	leaq	_ZSt4cout(%rip), %rbp
	movl	$23, %edx
	leaq	.LC5(%rip), %rsi
	movq	%rbp, %rdi
	vmovsd	%xmm2, 24(%rsp)
.LEHB0:
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	vmovsd	8(%rsp), %xmm0
	movq	%rbp, %rdi
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	movq	%rax, %rdi
	movl	$3, %edx
	leaq	.LC6(%rip), %rsi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movl	$23, %edx
	leaq	.LC7(%rip), %rsi
	movq	%rbp, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	vmovq	%r15, %xmm0
	movq	%rbp, %rdi
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	movq	%rax, %rdi
	movl	$7, %edx
	leaq	.LC8(%rip), %rsi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movl	$23, %edx
	leaq	.LC9(%rip), %rsi
	movq	%rbp, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	vmovsd	16(%rsp), %xmm0
	movq	%rbp, %rdi
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	movq	%rax, %rdi
	movl	$9, %edx
	leaq	.LC10(%rip), %rsi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movl	$23, %edx
	leaq	.LC11(%rip), %rsi
	movq	%rbp, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%r12, %rsi
	movq	%rbp, %rdi
	call	_ZNSo9_M_insertIyEERSoT_@PLT
	leaq	.LC12(%rip), %r12
	movq	%rax, %rdi
	movl	$1, %edx
	movq	%r12, %rsi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movl	$23, %edx
	leaq	.LC13(%rip), %rsi
	movq	%rbp, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	vmovsd	24(%rsp), %xmm0
	movq	%rbp, %rdi
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	movq	%rax, %rdi
	movl	$5, %edx
	leaq	.LC14(%rip), %rsi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movl	$23, %edx
	leaq	.LC15(%rip), %rsi
	movq	%rbp, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	96(%rsp), %rax
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	%rbp, %rdi
	vcvtss2sd	40(%rax), %xmm6, %xmm0
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	movq	%rax, %rdi
	movl	$1, %edx
	movq	%r12, %rsi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.LEHE0:
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	movq	%r13, %rsi
	movq	%rbx, 56(%rsp)
	leaq	_ZN6Matrix11storeResultEPf._omp_fn.0(%rip), %rdi
	movq	%r14, 48(%rsp)
	call	GOMP_parallel@PLT
	movq	88(%rsp), %rdi
	call	free@PLT
	movq	96(%rsp), %rdi
	call	free@PLT
	movq	104(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L172
	addq	$120, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L164:
	.cfi_restore_state
	movq	%r12, %rax
	movq	%r12, %rdx
	vxorpd	%xmm7, %xmm7, %xmm7
	shrq	%rax
	andl	$1, %edx
	orq	%rdx, %rax
	vcvtsi2sdq	%rax, %xmm7, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm0
	jmp	.L165
.L172:
	call	__stack_chk_fail@PLT
.L168:
	endbr64
	movq	%rax, %rbp
	jmp	.L166
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA8024:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE8024-.LLSDACSB8024
.LLSDACSB8024:
	.uleb128 .LEHB0-.LFB8024
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L168-.LFB8024
	.uleb128 0
.LLSDACSE8024:
	.text
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC8024
	.type	_Z9correlateiiPKfPf.cold, @function
_Z9correlateiiPKfPf.cold:
.LFSB8024:
.L166:
	.cfi_def_cfa_offset 176
	.cfi_offset 3, -56
	.cfi_offset 6, -48
	.cfi_offset 12, -40
	.cfi_offset 13, -32
	.cfi_offset 14, -24
	.cfi_offset 15, -16
	movq	88(%rsp), %rdi
	vzeroupper
	call	free@PLT
	movq	96(%rsp), %rdi
	call	free@PLT
	movq	%rbp, %rdi
.LEHB1:
	call	_Unwind_Resume@PLT
.LEHE1:
	.cfi_endproc
.LFE8024:
	.section	.gcc_except_table
.LLSDAC8024:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC8024-.LLSDACSBC8024
.LLSDACSBC8024:
	.uleb128 .LEHB1-.LCOLDB16
	.uleb128 .LEHE1-.LEHB1
	.uleb128 0
	.uleb128 0
.LLSDACSEC8024:
	.section	.text.unlikely
	.text
	.size	_Z9correlateiiPKfPf, .-_Z9correlateiiPKfPf
	.section	.text.unlikely
	.size	_Z9correlateiiPKfPf.cold, .-_Z9correlateiiPKfPf.cold
.LCOLDE16:
	.text
.LHOTE16:
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB8034:
	.cfi_startproc
	endbr64
	pushq	%r14
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	movl	$784000000, %edi
	pushq	%r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	call	malloc@PLT
	movl	$784000000, %edi
	movq	%rax, %r12
	call	malloc@PLT
	leaq	56000(%r12), %rbx
	leaq	784056000(%r12), %rbp
	movq	%rax, %r13
.L174:
	leaq	-56000(%rbx), %r14
	.p2align 4,,10
	.p2align 3
.L175:
	call	rand@PLT
	vxorps	%xmm1, %xmm1, %xmm1
	addq	$4, %r14
	vmovss	.LC19(%rip), %xmm2
	vcvtsi2ssl	%eax, %xmm1, %xmm0
	vmulss	.LC17(%rip), %xmm0, %xmm0
	vfmadd132ss	.LC18(%rip), %xmm2, %xmm0
	vmovss	%xmm0, -4(%r14)
	cmpq	%rbx, %r14
	jne	.L175
	leaq	56000(%r14), %rbx
	cmpq	%rbp, %rbx
	jne	.L174
	movq	%r13, %rcx
	movq	%r12, %rdx
	movl	$14000, %esi
	movl	$14000, %edi
	call	_Z9correlateiiPKfPf
	movq	%r13, %rdi
	call	free@PLT
	movq	%r12, %rdi
	call	free@PLT
	popq	%rbx
	.cfi_def_cfa_offset 40
	xorl	%eax, %eax
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE8034:
	.size	main, .-main
	.p2align 4
	.type	_GLOBAL__sub_I__Z9correlateiiPKfPf, @function
_GLOBAL__sub_I__Z9correlateiiPKfPf:
.LFB8707:
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
.LFE8707:
	.size	_GLOBAL__sub_I__Z9correlateiiPKfPf, .-_GLOBAL__sub_I__Z9correlateiiPKfPf
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I__Z9correlateiiPKfPf
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC1:
	.long	0
	.long	0
	.align 8
.LC2:
	.long	0
	.long	1072693248
	.align 8
.LC3:
	.long	0
	.long	1104006501
	.align 8
.LC4:
	.long	-400107883
	.long	1041313291
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC17:
	.long	805306368
	.align 4
.LC18:
	.long	1073741824
	.align 4
.LC19:
	.long	-1082130432
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
