
rng.o：     文件格式 elf64-littleriscv


Disassembly of section .text:

0000000000000000 <get_seed>:
   0:	7179                	add	sp,sp,-48
   2:	f406                	sd	ra,40(sp)
   4:	e84a                	sd	s2,16(sp)
   6:	e599                	bnez	a1,14 <.L2>

0000000000000008 <.L4>:
   8:	4905                	li	s2,1

000000000000000a <.L3>:
   a:	70a2                	ld	ra,40(sp)
   c:	854a                	mv	a0,s2
   e:	6942                	ld	s2,16(sp)
  10:	6145                	add	sp,sp,48
  12:	8082                	ret

0000000000000014 <.L2>:
  14:	f022                	sd	s0,32(sp)
  16:	ec26                	sd	s1,24(sp)
  18:	842e                	mv	s0,a1
  1a:	84aa                	mv	s1,a0
  1c:	00000097          	auipc	ra,0x0
  20:	000080e7          	jalr	ra # 1c <.L2+0x8>
  24:	cd15                	beqz	a0,60 <.L15>
  26:	4581                	li	a1,0
  28:	00000517          	auipc	a0,0x0
  2c:	00050513          	mv	a0,a0
  30:	e44e                	sd	s3,8(sp)
  32:	00000097          	auipc	ra,0x0
  36:	000080e7          	jalr	ra # 32 <.L2+0x1e>
  3a:	89aa                	mv	s3,a0
  3c:	4901                	li	s2,0
  3e:	02054463          	bltz	a0,66 <.L16>
  42:	4911                	li	s2,4

0000000000000044 <.L8>:
  44:	8622                	mv	a2,s0
  46:	85a6                	mv	a1,s1
  48:	854e                	mv	a0,s3
  4a:	00000097          	auipc	ra,0x0
  4e:	000080e7          	jalr	ra # 4a <.L8+0x6>
  52:	00054e63          	bltz	a0,6e <.L18>
  56:	8c09                	sub	s0,s0,a0
  58:	94aa                	add	s1,s1,a0
  5a:	f46d                	bnez	s0,44 <.L8>
  5c:	4905                	li	s2,1
  5e:	a005                	j	7e <.L7>

0000000000000060 <.L15>:
  60:	7402                	ld	s0,32(sp)
  62:	64e2                	ld	s1,24(sp)
  64:	b755                	j	8 <.L4>

0000000000000066 <.L16>:
  66:	7402                	ld	s0,32(sp)
  68:	64e2                	ld	s1,24(sp)
  6a:	69a2                	ld	s3,8(sp)
  6c:	bf79                	j	a <.L3>

000000000000006e <.L18>:
  6e:	00000097          	auipc	ra,0x0
  72:	000080e7          	jalr	ra # 6e <.L18>
  76:	411c                	lw	a5,0(a0)
  78:	fd2786e3          	beq	a5,s2,44 <.L8>
  7c:	4901                	li	s2,0

000000000000007e <.L7>:
  7e:	854e                	mv	a0,s3
  80:	00000097          	auipc	ra,0x0
  84:	000080e7          	jalr	ra # 80 <.L7+0x2>
  88:	70a2                	ld	ra,40(sp)
  8a:	7402                	ld	s0,32(sp)
  8c:	64e2                	ld	s1,24(sp)
  8e:	69a2                	ld	s3,8(sp)
  90:	854a                	mv	a0,s2
  92:	6942                	ld	s2,16(sp)
  94:	6145                	add	sp,sp,48
  96:	8082                	ret

0000000000000098 <prng_refill>:
  98:	6b2067b7          	lui	a5,0x6b206
  9c:	7109                	add	sp,sp,-384
  9e:	796236b7          	lui	a3,0x79623
  a2:	57478793          	add	a5,a5,1396 # 6b206574 <.L39+0x6b20605a>
  a6:	33206637          	lui	a2,0x33206
  aa:	ed3e                	sd	a5,152(sp)
  ac:	d3268793          	add	a5,a3,-718 # 79622d32 <.L39+0x79622818>
  b0:	f13e                	sd	a5,160(sp)
  b2:	00000697          	auipc	a3,0x0
  b6:	0006b683          	ld	a3,0(a3) # b2 <prng_refill+0x1a>
  ba:	46e60793          	add	a5,a2,1134 # 3320646e <.L39+0x33205f54>
  be:	f53e                	sd	a5,168(sp)
  c0:	629c                	ld	a5,0(a3)
  c2:	e63e                	sd	a5,264(sp)
  c4:	4781                	li	a5,0
  c6:	02050793          	add	a5,a0,32 # 48 <.L8+0x4>
  ca:	faa2                	sd	s0,368(sp)
  cc:	23853403          	ld	s0,568(a0)
  d0:	fe86                	sd	ra,376(sp)
  d2:	e13e                	sd	a5,128(sp)
  d4:	617080b7          	lui	ra,0x61708
  d8:	09bc                	add	a5,sp,216
  da:	86508093          	add	ra,ra,-1947 # 61707865 <.L39+0x6170734b>
  de:	e53e                	sd	a5,136(sp)
  e0:	20850793          	add	a5,a0,520
  e4:	f6a6                	sd	s1,360(sp)
  e6:	f2ca                	sd	s2,352(sp)
  e8:	eece                	sd	s3,344(sp)
  ea:	ead2                	sd	s4,336(sp)
  ec:	e6d6                	sd	s5,328(sp)
  ee:	e2da                	sd	s6,320(sp)
  f0:	fe5e                	sd	s7,312(sp)
  f2:	fa62                	sd	s8,304(sp)
  f4:	f666                	sd	s9,296(sp)
  f6:	f26a                	sd	s10,288(sp)
  f8:	ee6e                	sd	s11,280(sp)
  fa:	fcaa                	sd	a0,120(sp)
  fc:	f0aa                	sd	a0,96(sp)
  fe:	eca2                	sd	s0,88(sp)
 100:	e93e                	sd	a5,144(sp)
 102:	fd22                	sd	s0,184(sp)
 104:	f906                	sd	ra,176(sp)

0000000000000106 <.L21>:
 106:	65ca                	ld	a1,144(sp)
 108:	652a                	ld	a0,136(sp)
 10a:	03000613          	li	a2,48
 10e:	4ba9                	li	s7,10
 110:	00000097          	auipc	ra,0x0
 114:	000080e7          	jalr	ra # 110 <.L21+0xa>
 118:	57ee                	lw	a5,248(sp)
 11a:	6766                	ld	a4,88(sp)
 11c:	10012883          	lw	a7,256(sp)
 120:	ec3e                	sd	a5,24(sp)
 122:	57fe                	lw	a5,252(sp)
 124:	10412f03          	lw	t5,260(sp)
 128:	53be                	lw	t2,236(sp)
 12a:	f83e                	sd	a5,48(sp)
 12c:	57ce                	lw	a5,240(sp)
 12e:	436e                	lw	t1,216(sp)
 130:	54ae                	lw	s1,232(sp)
 132:	e43e                	sd	a5,8(sp)
 134:	57de                	lw	a5,244(sp)
 136:	45fe                	lw	a1,220(sp)
 138:	568e                	lw	a3,224(sp)
 13a:	f03e                	sd	a5,32(sp)
 13c:	0007079b          	sext.w	a5,a4
 140:	f8be                	sd	a5,112(sp)
 142:	00f8c7b3          	xor	a5,a7,a5
 146:	e0be                	sd	a5,64(sp)
 148:	67ea                	ld	a5,152(sp)
 14a:	9701                	sra	a4,a4,0x20
 14c:	5e9e                	lw	t4,228(sp)
 14e:	fc3e                	sd	a5,56(sp)
 150:	778a                	ld	a5,160(sp)
 152:	7a4a                	ld	s4,176(sp)
 154:	00ef4f33          	xor	t5,t5,a4
 158:	f43e                	sd	a5,40(sp)
 15a:	77aa                	ld	a5,168(sp)
 15c:	f4ba                	sd	a4,104(sp)
 15e:	4dc1                	li	s11,16
 160:	e83e                	sd	a5,16(sp)
 162:	4d31                	li	s10,12
 164:	4ca1                	li	s9,8
 166:	4c1d                	li	s8,7
 168:	8b7a                	mv	s6,t5
 16a:	e49e                	sd	t2,72(sp)

000000000000016c <.L20>:
 16c:	67c2                	ld	a5,16(sp)
 16e:	006a0a3b          	addw	s4,s4,t1
 172:	6626                	ld	a2,72(sp)
 174:	00b788bb          	addw	a7,a5,a1
 178:	77a2                	ld	a5,40(sp)
 17a:	6706                	ld	a4,64(sp)
 17c:	3bfd                	addw	s7,s7,-1
 17e:	00d78f3b          	addw	t5,a5,a3
 182:	77e2                	ld	a5,56(sp)
 184:	01e74833          	xor	a6,a4,t5
 188:	61b8183b          	rolw	a6,a6,s11
 18c:	01d78fbb          	addw	t6,a5,t4
 190:	67e2                	ld	a5,24(sp)
 192:	01fb4733          	xor	a4,s6,t6
 196:	61b7173b          	rolw	a4,a4,s11
 19a:	0147cab3          	xor	s5,a5,s4
 19e:	77c2                	ld	a5,48(sp)
 1a0:	61ba9abb          	rolw	s5,s5,s11
 1a4:	009a843b          	addw	s0,s5,s1
 1a8:	0117c7b3          	xor	a5,a5,a7
 1ac:	61b797bb          	rolw	a5,a5,s11
 1b0:	00c782bb          	addw	t0,a5,a2
 1b4:	6622                	ld	a2,8(sp)
 1b6:	0055c533          	xor	a0,a1,t0
 1ba:	61a5153b          	rolw	a0,a0,s10
 1be:	00c809bb          	addw	s3,a6,a2
 1c2:	7602                	ld	a2,32(sp)
 1c4:	00a888bb          	addw	a7,a7,a0
 1c8:	00834e33          	xor	t3,t1,s0
 1cc:	00c7093b          	addw	s2,a4,a2
 1d0:	0136c633          	xor	a2,a3,s3
 1d4:	012ec6b3          	xor	a3,t4,s2
 1d8:	61a696bb          	rolw	a3,a3,s10
 1dc:	00df8fbb          	addw	t6,t6,a3
 1e0:	01f745b3          	xor	a1,a4,t6
 1e4:	0117c7b3          	xor	a5,a5,a7
 1e8:	619595bb          	rolw	a1,a1,s9
 1ec:	00b9093b          	addw	s2,s2,a1
 1f0:	0126c6b3          	xor	a3,a3,s2
 1f4:	61ae1e3b          	rolw	t3,t3,s10
 1f8:	61a6163b          	rolw	a2,a2,s10
 1fc:	01ca0a3b          	addw	s4,s4,t3
 200:	00cf0f3b          	addw	t5,t5,a2
 204:	619797bb          	rolw	a5,a5,s9
 208:	00f282bb          	addw	t0,t0,a5
 20c:	014acab3          	xor	s5,s5,s4
 210:	01e84833          	xor	a6,a6,t5
 214:	00554533          	xor	a0,a0,t0
 218:	618696bb          	rolw	a3,a3,s8
 21c:	00df0f3b          	addw	t5,t5,a3
 220:	619a9abb          	rolw	s5,s5,s9
 224:	01e7c7b3          	xor	a5,a5,t5
 228:	0154043b          	addw	s0,s0,s5
 22c:	6185153b          	rolw	a0,a0,s8
 230:	00aa0a3b          	addw	s4,s4,a0
 234:	008e4e33          	xor	t3,t3,s0
 238:	0145c5b3          	xor	a1,a1,s4
 23c:	6198183b          	rolw	a6,a6,s9
 240:	61b797bb          	rolw	a5,a5,s11
 244:	010989bb          	addw	s3,s3,a6
 248:	9c3d                	addw	s0,s0,a5
 24a:	01364633          	xor	a2,a2,s3
 24e:	8ea1                	xor	a3,a3,s0
 250:	61b595bb          	rolw	a1,a1,s11
 254:	00b989bb          	addw	s3,s3,a1
 258:	01354533          	xor	a0,a0,s3
 25c:	618e1e3b          	rolw	t3,t3,s8
 260:	6186163b          	rolw	a2,a2,s8
 264:	01fe0fbb          	addw	t6,t3,t6
 268:	00c888bb          	addw	a7,a7,a2
 26c:	61a696bb          	rolw	a3,a3,s10
 270:	00df0f3b          	addw	t5,t5,a3
 274:	011ac733          	xor	a4,s5,a7
 278:	01f84833          	xor	a6,a6,t6
 27c:	61a5153b          	rolw	a0,a0,s10
 280:	01e7c7b3          	xor	a5,a5,t5
 284:	00aa0abb          	addw	s5,s4,a0
 288:	0155c5b3          	xor	a1,a1,s5
 28c:	61b7173b          	rolw	a4,a4,s11
 290:	61b8183b          	rolw	a6,a6,s11
 294:	00e9093b          	addw	s2,s2,a4
 298:	010282bb          	addw	t0,t0,a6
 29c:	619797bb          	rolw	a5,a5,s9
 2a0:	9c3d                	addw	s0,s0,a5
 2a2:	61959b3b          	rolw	s6,a1,s9
 2a6:	01264633          	xor	a2,a2,s2
 2aa:	005e4e33          	xor	t3,t3,t0
 2ae:	016989bb          	addw	s3,s3,s6
 2b2:	8ea1                	xor	a3,a3,s0
 2b4:	61a6163b          	rolw	a2,a2,s10
 2b8:	61ae1e3b          	rolw	t3,t3,s10
 2bc:	00c883bb          	addw	t2,a7,a2
 2c0:	01cf8fbb          	addw	t6,t6,t3
 2c4:	61869ebb          	rolw	t4,a3,s8
 2c8:	0009869b          	sext.w	a3,s3
 2cc:	000f831b          	sext.w	t1,t6
 2d0:	00774733          	xor	a4,a4,t2
 2d4:	01f84833          	xor	a6,a6,t6
 2d8:	e436                	sd	a3,8(sp)
 2da:	0003869b          	sext.w	a3,t2
 2de:	000f049b          	sext.w	s1,t5
 2e2:	6197173b          	rolw	a4,a4,s9
 2e6:	6198183b          	rolw	a6,a6,s9
 2ea:	00e9093b          	addw	s2,s2,a4
 2ee:	010282bb          	addw	t0,t0,a6
 2f2:	e836                	sd	a3,16(sp)
 2f4:	fc1a                	sd	t1,56(sp)
 2f6:	0007069b          	sext.w	a3,a4
 2fa:	0008031b          	sext.w	t1,a6
 2fe:	ec36                	sd	a3,24(sp)
 300:	f426                	sd	s1,40(sp)
 302:	0009069b          	sext.w	a3,s2
 306:	0007849b          	sext.w	s1,a5
 30a:	e09a                	sd	t1,64(sp)
 30c:	0002831b          	sext.w	t1,t0
 310:	01354533          	xor	a0,a0,s3
 314:	01264633          	xor	a2,a2,s2
 318:	005e4e33          	xor	t3,t3,t0
 31c:	f036                	sd	a3,32(sp)
 31e:	f826                	sd	s1,48(sp)
 320:	e49a                	sd	t1,72(sp)
 322:	caf6                	sw	t4,84(sp)
 324:	618515bb          	rolw	a1,a0,s8
 328:	6186163b          	rolw	a2,a2,s8
 32c:	618e1e3b          	rolw	t3,t3,s8
 330:	88da                	mv	a7,s6
 332:	852e                	mv	a0,a1
 334:	000a8a1b          	sext.w	s4,s5
 338:	0006069b          	sext.w	a3,a2
 33c:	0004049b          	sext.w	s1,s0
 340:	000e031b          	sext.w	t1,t3
 344:	e20b94e3          	bnez	s7,16c <.L20>
 348:	7c66                	ld	s8,120(sp)
 34a:	c43e                	sw	a5,8(sp)
 34c:	77c6                	ld	a5,112(sp)
 34e:	230c2a03          	lw	s4,560(s8)
 352:	234c2483          	lw	s1,564(s8)
 356:	224c2683          	lw	a3,548(s8)
 35a:	00fa4a33          	xor	s4,s4,a5
 35e:	77a6                	ld	a5,104(sp)
 360:	00d9093b          	addw	s2,s2,a3
 364:	8bfe                	mv	s7,t6
 366:	8cbd                	xor	s1,s1,a5
 368:	011486bb          	addw	a3,s1,a7
 36c:	332064b7          	lui	s1,0x33206
 370:	8fd6                	mv	t6,s5
 372:	46e4849b          	addw	s1,s1,1134 # 3320646e <.L39+0x33205f54>
 376:	8a9e                	mv	s5,t2
 378:	010a0a3b          	addw	s4,s4,a6
 37c:	009a883b          	addw	a6,s5,s1
 380:	796234b7          	lui	s1,0x79623
 384:	218c2303          	lw	t1,536(s8)
 388:	220c2583          	lw	a1,544(s8)
 38c:	214c2e83          	lw	t4,532(s8)
 390:	47d6                	lw	a5,84(sp)
 392:	8b7a                	mv	s6,t5
 394:	d324849b          	addw	s1,s1,-718 # 79622d32 <.L39+0x79622818>
 398:	009b08bb          	addw	a7,s6,s1
 39c:	6b2064b7          	lui	s1,0x6b206
 3a0:	208c2383          	lw	t2,520(s8)
 3a4:	20cc2d03          	lw	s10,524(s8)
 3a8:	210c2f03          	lw	t5,528(s8)
 3ac:	21cc2c83          	lw	s9,540(s8)
 3b0:	228c2d83          	lw	s11,552(s8)
 3b4:	5744849b          	addw	s1,s1,1396 # 6b206574 <.L39+0x6b20605a>
 3b8:	22cc2c03          	lw	s8,556(s8)
 3bc:	01d78ebb          	addw	t4,a5,t4
 3c0:	0064043b          	addw	s0,s0,t1
 3c4:	47a2                	lw	a5,8(sp)
 3c6:	00b989bb          	addw	s3,s3,a1
 3ca:	009b833b          	addw	t1,s7,s1
 3ce:	617085b7          	lui	a1,0x61708
 3d2:	7486                	ld	s1,96(sp)
 3d4:	8655859b          	addw	a1,a1,-1947 # 61707865 <.L39+0x6170734b>
 3d8:	01b7073b          	addw	a4,a4,s11
 3dc:	007e0e3b          	addw	t3,t3,t2
 3e0:	01a5053b          	addw	a0,a0,s10
 3e4:	01e6063b          	addw	a2,a2,t5
 3e8:	019282bb          	addw	t0,t0,s9
 3ec:	00bf85bb          	addw	a1,t6,a1
 3f0:	018787bb          	addw	a5,a5,s8
 3f4:	18e4a023          	sw	a4,384(s1)
 3f8:	c08c                	sw	a1,0(s1)
 3fa:	0304a023          	sw	a6,32(s1)
 3fe:	0514a023          	sw	a7,64(s1)
 402:	0664a023          	sw	t1,96(s1)
 406:	09c4a023          	sw	t3,128(s1)
 40a:	0aa4a023          	sw	a0,160(s1)
 40e:	0cc4a023          	sw	a2,192(s1)
 412:	0fd4a023          	sw	t4,224(s1)
 416:	1084a023          	sw	s0,256(s1)
 41a:	1254a023          	sw	t0,288(s1)
 41e:	1534a023          	sw	s3,320(s1)
 422:	1724a023          	sw	s2,352(s1)
 426:	1af4a023          	sw	a5,416(s1)
 42a:	6766                	ld	a4,88(sp)
 42c:	00448793          	add	a5,s1,4
 430:	1d44a023          	sw	s4,448(s1)
 434:	0705                	add	a4,a4,1
 436:	ecba                	sd	a4,88(sp)
 438:	670a                	ld	a4,128(sp)
 43a:	1ed4a023          	sw	a3,480(s1)
 43e:	f0be                	sd	a5,96(sp)
 440:	cce793e3          	bne	a5,a4,106 <.L21>
 444:	746a                	ld	s0,184(sp)
 446:	77e6                	ld	a5,120(sp)
 448:	0421                	add	s0,s0,8
 44a:	2287bc23          	sd	s0,568(a5)
 44e:	2007b023          	sd	zero,512(a5)
 452:	00000797          	auipc	a5,0x0
 456:	0007b783          	ld	a5,0(a5) # 452 <.L20+0x2e6>
 45a:	6732                	ld	a4,264(sp)
 45c:	639c                	ld	a5,0(a5)
 45e:	8fb9                	xor	a5,a5,a4
 460:	4701                	li	a4,0
 462:	e385                	bnez	a5,482 <.L26>
 464:	70f6                	ld	ra,376(sp)
 466:	7456                	ld	s0,368(sp)
 468:	74b6                	ld	s1,360(sp)
 46a:	7916                	ld	s2,352(sp)
 46c:	69f6                	ld	s3,344(sp)
 46e:	6a56                	ld	s4,336(sp)
 470:	6ab6                	ld	s5,328(sp)
 472:	6b16                	ld	s6,320(sp)
 474:	7bf2                	ld	s7,312(sp)
 476:	7c52                	ld	s8,304(sp)
 478:	7cb2                	ld	s9,296(sp)
 47a:	7d12                	ld	s10,288(sp)
 47c:	6df2                	ld	s11,280(sp)
 47e:	6119                	add	sp,sp,384
 480:	8082                	ret

0000000000000482 <.L26>:
 482:	00000097          	auipc	ra,0x0
 486:	000080e7          	jalr	ra # 482 <.L26>

000000000000048a <prng_init>:
 48a:	1141                	add	sp,sp,-16
 48c:	e022                	sd	s0,0(sp)
 48e:	842a                	mv	s0,a0
 490:	03800613          	li	a2,56
 494:	852e                	mv	a0,a1
 496:	20840593          	add	a1,s0,520
 49a:	e406                	sd	ra,8(sp)
 49c:	00000097          	auipc	ra,0x0
 4a0:	000080e7          	jalr	ra # 49c <prng_init+0x12>
 4a4:	8522                	mv	a0,s0
 4a6:	6402                	ld	s0,0(sp)
 4a8:	60a2                	ld	ra,8(sp)
 4aa:	0141                	add	sp,sp,16
 4ac:	00000317          	auipc	t1,0x0
 4b0:	00030067          	jr	t1 # 4ac <prng_init+0x22>

00000000000004b4 <prng_get_bytes>:
 4b4:	c23d                	beqz	a2,51a <.L39>
 4b6:	7179                	add	sp,sp,-48
 4b8:	ec26                	sd	s1,24(sp)
 4ba:	e84a                	sd	s2,16(sp)
 4bc:	e44e                	sd	s3,8(sp)
 4be:	e052                	sd	s4,0(sp)
 4c0:	f406                	sd	ra,40(sp)
 4c2:	f022                	sd	s0,32(sp)
 4c4:	84b2                	mv	s1,a2
 4c6:	892a                	mv	s2,a0
 4c8:	89ae                	mv	s3,a1
 4ca:	20000a13          	li	s4,512

00000000000004ce <.L33>:
 4ce:	20093403          	ld	s0,512(s2)
 4d2:	854e                	mv	a0,s3
 4d4:	85ca                	mv	a1,s2
 4d6:	408a0433          	sub	s0,s4,s0
 4da:	0a945433          	minu	s0,s0,s1
 4de:	8622                	mv	a2,s0
 4e0:	00000097          	auipc	ra,0x0
 4e4:	000080e7          	jalr	ra # 4e0 <.L33+0x12>
 4e8:	20093783          	ld	a5,512(s2)
 4ec:	99a2                	add	s3,s3,s0
 4ee:	8c81                	sub	s1,s1,s0
 4f0:	943e                	add	s0,s0,a5
 4f2:	01440d63          	beq	s0,s4,50c <.L31>
 4f6:	20893023          	sd	s0,512(s2)
 4fa:	f8f1                	bnez	s1,4ce <.L33>

00000000000004fc <.L42>:
 4fc:	70a2                	ld	ra,40(sp)
 4fe:	7402                	ld	s0,32(sp)
 500:	64e2                	ld	s1,24(sp)
 502:	6942                	ld	s2,16(sp)
 504:	69a2                	ld	s3,8(sp)
 506:	6a02                	ld	s4,0(sp)
 508:	6145                	add	sp,sp,48
 50a:	8082                	ret

000000000000050c <.L31>:
 50c:	854a                	mv	a0,s2
 50e:	00000097          	auipc	ra,0x0
 512:	000080e7          	jalr	ra # 50e <.L31+0x2>
 516:	fcc5                	bnez	s1,4ce <.L33>
 518:	b7d5                	j	4fc <.L42>

000000000000051a <.L39>:
 51a:	8082                	ret

Disassembly of section .rodata.str1.8:

0000000000000000 <.LC0>:
   0:	7665642f          	.insn	4, 0x7665642f
   4:	6172752f          	.insn	4, 0x6172752f
   8:	646e                	ld	s0,216(sp)
   a:	Address 0xa is out of bounds.


Disassembly of section .comment:

0000000000000000 <.comment>:
   0:	4700                	lw	s0,8(a4)
   2:	203a4343          	fmadd.s	ft6,fs4,ft3,ft4,rmm
   6:	5528                	lw	a0,104(a0)
   8:	7562                	ld	a0,56(sp)
   a:	746e                	ld	s0,248(sp)
   c:	2075                	.insn	2, 0x2075
   e:	3331                	addw	t1,t1,-20
  10:	322e                	fld	ft4,232(sp)
  12:	302e                	fld	ft0,232(sp)
  14:	342d                	addw	s0,s0,-21
  16:	6275                	lui	tp,0x1d
  18:	6e75                	lui	t3,0x1d
  1a:	7574                	ld	a3,232(a0)
  1c:	62622d33          	.insn	4, 0x62622d33
  20:	2932                	fld	fs2,264(sp)
  22:	3120                	fld	fs0,96(a0)
  24:	2e322e33          	.insn	4, 0x2e322e33
  28:	0030                	add	a2,sp,8

Disassembly of section .eh_frame:

0000000000000000 <.eh_frame>:
   0:	0010                	.insn	2, 0x0010
   2:	0000                	unimp
   4:	0000                	unimp
   6:	0000                	unimp
   8:	00527a03          	.insn	4, 0x00527a03
   c:	7c01                	lui	s8,0xfffe0
   e:	0101                	add	sp,sp,0
  10:	00020d1b          	sext.w	s10,tp
  14:	004c                	add	a1,sp,4
  16:	0000                	unimp
  18:	0018                	.insn	2, 0x0018
	...
  22:	0000                	unimp
  24:	4200                	lw	s0,0(a2)
  26:	300e                	fld	ft0,224(sp)
  28:	8144                	.insn	2, 0x8144
  2a:	9202                	jalr	tp
  2c:	4608                	lw	a0,8(a2)
  2e:	c10a                	sw	sp,128(sp)
  30:	d244                	sw	s1,36(a2)
  32:	0e42                	sll	t3,t3,0x10
  34:	4200                	lw	s0,0(a2)
  36:	0488440b          	.insn	4, 0x0488440b
  3a:	0689                	add	a3,a3,2
  3c:	935a                	add	t1,t1,s6
  3e:	6e0a                	ld	t3,128(sp)
  40:	42c842d3          	.insn	4, 0x42c842d3
  44:	42c9                	li	t0,18
  46:	0488                	add	a0,sp,576
  48:	0689                	add	a3,a3,2
  4a:	0a420a93          	add	s5,tp,164 # 1d0a4 <.L39+0x1cb8a>
  4e:	42c8                	lw	a0,4(a3)
  50:	42c9                	li	t0,18
  52:	5c0b42d3          	.insn	4, 0x5c0b42d3
  56:	42c1                	li	t0,16
  58:	42c8                	lw	a0,4(a3)
  5a:	42c9                	li	t0,18
  5c:	42d244d3          	.insn	4, 0x42d244d3
  60:	000e                	c.slli	zero,0x3
  62:	0000                	unimp
  64:	0050                	add	a2,sp,4
  66:	0000                	unimp
  68:	0068                	add	a0,sp,12
	...
  72:	0000                	unimp
  74:	4600                	lw	s0,8(a2)
  76:	800e                	c.mv	zero,gp
  78:	04886e03          	lwu	t3,72(a6)
  7c:	8148                	.insn	2, 0x8148
  7e:	6602                	ld	a2,0(sp)
  80:	0689                	add	a3,a3,2
  82:	0892                	sll	a7,a7,0x4
  84:	0c940a93          	add	s5,s0,201
  88:	0e95                	add	t4,t4,5
  8a:	1096                	sll	ra,ra,0x25
  8c:	14981297          	auipc	t0,0x14981
  90:	1699                	add	a3,a3,-26
  92:	189a                	sll	a7,a7,0x26
  94:	6c031a9b          	.insn	4, 0x6c031a9b
  98:	42c10a03          	lb	s4,1068(sp)
  9c:	42c8                	lw	a0,4(a3)
  9e:	42c9                	li	t0,18
  a0:	42d2                	lw	t0,20(sp)
  a2:	42d442d3          	.insn	4, 0x42d442d3
  a6:	42d5                	li	t0,21
  a8:	42d6                	lw	t0,84(sp)
  aa:	42d842d7          	.insn	4, 0x42d842d7
  ae:	42d9                	li	t0,22
  b0:	42da                	lw	t0,148(sp)
  b2:	000e42db          	.insn	4, 0x000e42db
  b6:	0b42                	sll	s6,s6,0x10
  b8:	0020                	add	s0,sp,8
  ba:	0000                	unimp
  bc:	00bc                	add	a5,sp,72
	...
  c6:	0000                	unimp
  c8:	4200                	lw	s0,0(a2)
  ca:	100e                	c.slli	zero,0x23
  cc:	8842                	mv	a6,a6
  ce:	4e04                	lw	s1,24(a2)
  d0:	0281                	add	t0,t0,0 # 1498108c <.L39+0x14980b72>
  d2:	c84c                	sw	a1,20(s0)
  d4:	c142                	sw	a6,128(sp)
  d6:	0e42                	sll	t3,t3,0x10
  d8:	0000                	unimp
  da:	0000                	unimp
  dc:	0038                	add	a4,sp,8
  de:	0000                	unimp
  e0:	00e0                	add	s0,sp,76
	...
  ea:	0000                	unimp
  ec:	4400                	lw	s0,8(s0)
  ee:	300e                	fld	ft0,224(sp)
  f0:	894c                	.insn	2, 0x894c
  f2:	9206                	add	tp,tp,ra
  f4:	9308                	.insn	2, 0x9308
  f6:	940a                	add	s0,s0,sp
  f8:	810c                	.insn	2, 0x810c
  fa:	8802                	jr	a6
  fc:	7a04                	ld	s1,48(a2)
  fe:	c10a                	sw	sp,128(sp)
 100:	c842                	sw	a6,16(sp)
 102:	c942                	sw	a6,144(sp)
 104:	d242                	sw	a6,36(sp)
 106:	d342                	sw	a6,164(sp)
 108:	d442                	sw	a6,40(sp)
 10a:	0e42                	sll	t3,t3,0x10
 10c:	4200                	lw	s0,0(a2)
 10e:	000e4e0b          	.insn	4, 0x000e4e0b
 112:	c8c1                	beqz	s1,1a2 <.L20+0x36>
 114:	d2c9                	beqz	a3,96 <.eh_frame+0x96>
 116:	Address 0x116 is out of bounds.


Disassembly of section .riscv.attributes:

0000000000000000 <.riscv.attributes>:
   0:	b641                	j	fffffffffffffb80 <.L39+0xfffffffffffff666>
   2:	0000                	unimp
   4:	7200                	ld	s0,32(a2)
   6:	7369                	lui	t1,0xffffa
   8:	01007663          	bgeu	zero,a6,14 <.riscv.attributes+0x14>
   c:	00ac                	add	a1,sp,72
   e:	0000                	unimp
  10:	1004                	add	s1,sp,32
  12:	7205                	lui	tp,0xfffe1
  14:	3676                	fld	fa2,376(sp)
  16:	6934                	ld	a3,80(a0)
  18:	7032                	.insn	2, 0x7032
  1a:	5f31                	li	t5,-20
  1c:	326d                	addw	tp,tp,-5 # fffffffffffe0ffb <.L39+0xfffffffffffe0ae1>
  1e:	3070                	fld	fa2,224(s0)
  20:	615f 7032 5f31      	.insn	6, 0x5f317032615f
  26:	3266                	fld	ft4,120(sp)
  28:	3270                	fld	fa2,224(a2)
  2a:	645f 7032 5f32      	.insn	6, 0x5f327032645f
  30:	30703263          	.insn	4, 0x30703263
  34:	765f 7031 5f30      	.insn	6, 0x5f307031765f
  3a:	697a                	ld	s2,408(sp)
  3c:	32727363          	bgeu	tp,t2,362 <.L20+0x1f6>
  40:	3070                	fld	fa2,224(s0)
  42:	7a5f 6669 6e65      	.insn	6, 0x6e6566697a5f
  48:	32696563          	bltu	s2,t1,372 <.L20+0x206>
  4c:	3070                	fld	fa2,224(s0)
  4e:	7a5f 6d6d 6c75      	.insn	6, 0x6c756d6d7a5f
  54:	7031                	c.lui	zero,0xfffec
  56:	5f30                	lw	a2,120(a4)
  58:	627a                	ld	tp,408(sp)
  5a:	3161                	addw	sp,sp,-8
  5c:	3070                	fld	fa2,224(s0)
  5e:	7a5f 6262 7031      	.insn	6, 0x703162627a5f
  64:	5f30                	lw	a2,120(a4)
  66:	767a                	ld	a2,440(sp)
  68:	3365                	addw	t1,t1,-7 # ffffffffffff9ff9 <.L39+0xffffffffffff9adf>
  6a:	6632                	ld	a2,264(sp)
  6c:	7031                	c.lui	zero,0xfffec
  6e:	5f30                	lw	a2,120(a4)
  70:	767a                	ld	a2,440(sp)
  72:	3365                	addw	t1,t1,-7
  74:	7832                	ld	a6,296(sp)
  76:	7031                	c.lui	zero,0xfffec
  78:	5f30                	lw	a2,120(a4)
  7a:	767a                	ld	a2,440(sp)
  7c:	3665                	addw	a2,a2,-7
  7e:	6434                	ld	a3,72(s0)
  80:	7031                	c.lui	zero,0xfffec
  82:	5f30                	lw	a2,120(a4)
  84:	767a                	ld	a2,440(sp)
  86:	3665                	addw	a2,a2,-7
  88:	6634                	ld	a3,72(a2)
  8a:	7031                	c.lui	zero,0xfffec
  8c:	5f30                	lw	a2,120(a4)
  8e:	767a                	ld	a2,440(sp)
  90:	3665                	addw	a2,a2,-7
  92:	7834                	ld	a3,112(s0)
  94:	7031                	c.lui	zero,0xfffec
  96:	5f30                	lw	a2,120(a4)
  98:	767a                	ld	a2,440(sp)
  9a:	316c                	fld	fa1,224(a0)
  9c:	3832                	fld	fa6,296(sp)
  9e:	3162                	fld	ft2,56(sp)
  a0:	3070                	fld	fa2,224(s0)
  a2:	7a5f 6c76 3233      	.insn	6, 0x32336c767a5f
  a8:	3162                	fld	ft2,56(sp)
  aa:	3070                	fld	fa2,224(s0)
  ac:	7a5f 6c76 3436      	.insn	6, 0x34366c767a5f
  b2:	3162                	fld	ft2,56(sp)
  b4:	3070                	fld	fa2,224(s0)
	...
