"""Real OmegaConf 2.3.0 metadata, decoded without importing OmegaConf.

Fixtures are compressed torch.save ZIP checkpoints created with OmegaConf
2.3.0, the exact version in CXR-CLIP's pinned requirements. Configs passed
through the official recursive DictConfig-only conversion. Their ListConfig
values still have DictConfig parents, nodes, flags, and resolver-cache data.
The synthetic model contains only a 2-by-3 arange tensor, never actual weights.
"""

import base64
import builtins
from collections import OrderedDict
from pathlib import Path
import pickle
import pickletools
import struct
import tempfile
import unittest
from unittest import mock
import zipfile
import zlib

import numpy as np
import torch

from foundation_sft.models.common import load_checkpoint_payload
from foundation_sft.models.cxr_clip import _preprocess, _training_config


_FIXTURES = {
    2: (
        'c-p0z349dQ9mhAsB(Ojr$Pp0{M07b4at9Cv5)H~?gu!D)ah&W<_HD99{%>Xp0iS{hf+u+2_kG{D)t+kWQSE7KYg*e1t*y0}S6lo3'
        '|Jm8RnN6^^`ysnC|9$WK`~Uv)pLw&x%5d>Mfxz(Lfr8iAz_@_Hqb#w`oEI}}qq?U%+164XSmSwZ8YUnRij>L;(M-BC5r-lH!$Oe~'
        '8AxSfW)g}8>=TMekwDr=nNSiLiDQ$A9zB}1>`V$uBco*`kuu_@Zl<Gh5{EJY!yT%X$?~WPBWN1S#^Z@}ywi}8AS;O+f+~q9&sdpG'
        'd!E&sNLSg3bYE4EVMkd*mE4bIdYh~2>Kb5VB<QB3+kHJIjG~-y4vePndbnwGBat>^P`*8zrLQ{-%Y-ok#?lBUM=P<xgo<##MKLUm'
        'PV-=#fPF)e5tNh6N{_DRFkZlf2v#@B%=JAcw_&1yNs%%qV%fC~P$^))EKQS`wAGp6DVQu^f2W8JdWwJpLXir&uNl%)N;5HwSS!!;'
        'z*GSTIt8Z8riJ@yq7^lgCQOfJQfA!1V^f_>ShjPJf-B@^tFs9Q&Dsnz1RU(7uBS}fKvOVNnTmA?$_X~shaa085`;tK+(QL~LgbN~'
        'olb3MMdgP)P{F0?z97t|W0&dZGNZQh{hX+-cVrVud9ZXnh<OEJE+H~Kb|RBDl2ElhQY`no%|57R)5%e1(ui9yPe2V-o_2Mf$=Fc4'
        'BT`QKRwlX5<a*SQ#f7?PG?Pq9QAxCdF|*UiChb@vD#hwO2{^Hat<b1J6Qv<#olKj(nP#LXm>(G>hnPt-Wu|k67O>gb4mzrOkHKXa'
        '7An<4ey;FG%1yWMhsokC5^#8k&16`}BLplCu~}?3o5LDd)pl0RrZQ=F39H?~>R3IK87|eJg*D3|q{tRBgtQ7c(uEu)pv?!Ff{<kj'
        '<Y*0!DS#{&5OyIe1RU#wOh!n%0y$2Dl?9Mh0yG!0TEOvMNINo4P#7m_aFWVsr{e3fRyNi!BH-i@D<<+90jId+Qw6N`rB{sT(-gF>'
        '!Rem#Rxyb)LaanNFll)?u&_zn<wg)XBs(e~CUyM1KhW}~9MmZwo=c$J(Jw)LrsyX$=px-UbZYABWKVaqQW+{EKS=>8*H2nN#^<LD'
        'c|8gbG~gaTYtWAsVj~!)Z41aE4zpV)px2w-2&Ao7Xnh)NAWD*Tv!L86A<9MpXSft0;LJfho~2MWX|Op@87a3$5#?+FTU^RH0=9Z7'
        'qmZ&qp`5G1_B>^@+$tx^4gu%6l=B5#;H8u!<wAvWkp>sL6p72yKQ@MHy;Hz0x7L>kxYSqcG03`1VO_4l6@|TXC99yst`cyyo7go1'
        'uJtBXfw1co*!3FRK#-lRJKUe`u;k7-O6x`eH@RuuEZ`PjTH}y(t3tX>gWC(!x+BEK6Yfp{ce%K`1>EDqjYr(Q3hq7)?)TtsMBD>%'
        'b0YB`6!4JCdsx6DKHfy+J*w~?)8O$!-V<`Ol6X%Fc*^BHE#Mg+uM&CBD!k`3c)pPLHRr%iX3J^0!u290mpu+IkjRSyzV3>AL%=tE'
        'A}^rGw-k|YYw(@In!ZHqO@}<^EqIyX0{M!7@4DRY33%1z&ULsiBlk6h`?>~iNbdYR_eB<23*Qe>2fZoaEw_VyAmE3d4hjd5^&^G#'
        'V-0>nj8|Cqvc(^~im%sX08u{`@V1NknSh^rQALP)M?w8UgI^AyKK%Igw<n%EC);6CThhfhzx=;T_3Rh$E7$k01-$30=esEI8%5x^'
        '8vL%Xp6`cfZA>O?lgsPiZYt>>0l#<Ae-QA2ucW&X{YM4;Ck;OIl(Zj9`e!P`M*{xhmf^1g{x(QO|E{3^p~1(5*u%Ob(`9A{Z?xsj'
        'n4Xe%q<Y%KTL^Q#8O_?}!fe_yI!yteWIGI5BD^I{%c1fn@l$f~nSjq-7ylISFQ1D~QTpGC^cNZr6v;bTzY$&}qaj4{VRB3GV%jG#'
        '--jY46jAZyjF%!WufWUXuEvM^MLt835rmKx<3U1>%!$al@lg~RJs_e~kC%Hj#*oHX`Ht)^%vfICikzn86+{}Ru=l0N_yRUID4*bE'
        'PbBsvFT0(Z6kC#45^F!DJd-K1zfztC1wX}$KY;L4o$}09%Om)KGD@euT&=SvpN50b&~y?wNHH{nA_w~ob<q?lI@2q92#Fr*da5K('
        'JLs%4^NW>CAVk0^tkumT5#zIPFdCmtnsXH6b170)V7zNqE{$q$8uKWP8o&4DC|!#b>Tl=A1+Q~;>q(_S(QTwiQ=x8MPPf^sJD+qH'
        '_;kC_5K1q^&uLWY!-%{{sr2C#Il^CQIbP~4_UbGlouzJ>>z%fA7G3(y&?;{$O^!dCc#CVVl?0Ad>>WjswnBR?IeW{zvPYBbF+<r~'
        'K433Q<Q0m&V=2-;xV_`NIx9(Mm20n6wI>^hZZz>a#x>X5YN8*ncsqe2Cl-3sa^6nzYDP%&<Uw0z4N|)DeR2w+PgTU$QslIu`$YGO'
        'olatgzmThQh3pt8WR%D;rI02?ItMRg+^fS#C*c<I6t$3D`NrrXVz=TaNs-jhjgj{1WJsrHC_gaZhZEUS{MZ!94(?~2SErYB*1LYX'
        'e2tO2xbzWgDqBPsm(~?ak3NcTaLsNcxib{Af+A-Yn%$5ydzLqeO_aoDf2Xvf^x1e}4Z070lJb_#=3CsW>p3K}Rq?)!BIg!dUAN?v'
        'w|kX$kn(x1{f%UQ*^;r&gDkfSn8*2qbV_~!iCn12UPO_L2QT?fujVe&yd<Z|7CH51)7cD?TFWoZFSVBu_Hsq&3W{7g_)>e7SLkXI'
        'y2cfnfj4e@uEy6jx#jj+qFtx(ucycjgD<x?diggI|7H*WBbSeh6CH2CZ;|hG0P=;kvb4?!erx_ZAoy+6)8T%L&I!S9clWb&g%SJ?'
        'W$>N8!FS~cXW0va-;KL3vsWzkuHg5Oo89u3?Nj!d{Jz-pbmur8Gr1YV$16d8uN*o{<M(0H;eys{CgRNI_sjjLo#9bdt!M2-(hBO_'
        '#D_aE-8R!!hU@F<g8Tv0aqd?wIoODs{K5RhU|TX{*sGm*kUxY9@-I-sAI435VioPl@<-%OxYXv4%Fk8dKzLZVcpHBVgT>*ZZTxY$'
        'wWBMS<r78tknc%43Qx&*OYo=B_e^)dGf2jI^Jl3@&*46%CvQ_NX`Vli;3CViI<nO(!)3$cYf>9_P6!0>Z{?8BcP9nr<e%?W<3mTo'
        ')^o(RmP37yd53(}qTrJszJrZ{_qBTwoE&H#5X_ZUmv=bvG(JabYpMTlHPu&B117Mz7xoO362tbIKt*72K7o!t+st6kw6z@ezv#Lz'
        '`IE*@nwZ1Z!{6Hvo2_)OIg|&E@H|zYSJT$A@c&4_p>A3%-<HUK@^6#;yJ-n#G->DiyDH@um3z%-jM4{R{rJ}hQR&Sa;wOB0{ig=%'
        '1{#3OTP9D)4%lGeNvtPgZ7tPbu@<$B&2{r@7c6M3t)1W4(9}?0Q`c0}SYK0L-#C9kQ(bd&U1L*2(}HHqvbSc}=OLL#h_WogWb=8d'
        '=gioG%>Dll;-9Ua0Fyj3`zHj3I?MbJwTUNtAgd+@_HyR<Icg(TcsOq<ocsykpRRUIxo7%`lLEfp_0LuNV2o$(Ze{Ke{?)@X)iYVE'
        '417uBtPGcy&UM?YxGF!h%=h_5k@NXUrSmE0(N{0!3hrFNN=ltMWi;Ly`yb;d+$8'
    ),
    3: (
        'c-p0z349dQ9mhAsB(Ojr$Pp1yL~uD0vN;GK3M3Mcg#d%c8pU<8JK49%9{InS1p+?B0|YO?`v&j(zOQ0WvGu67+S*#BwXM+FT6=l)'
        'eE<LKY~IW!sO^5p?#zGR`~Lp_-~4CZ?64wKe2~vKaG<Z?HQYDCr}HRFtTAT9bW5-9?n<^amHXOUuZ_h5e1UMOoDj{VI}&jy5-=bT'
        'E|Gy$CT1i-4mc<fmLk5io-&{$JQ&9&6Wx($*0eGyC=Cylkwi+58xbQNm6JG>2^eTo%}ka@4H!hzST-I{q~jgBjQCkecrvOaqC8_}'
        'I;<Jy+C;j_N~C+Lx^*kc>Z{~_EVH(;s<yTs28aDlN)fB4+khdI6V8F5^j$YMEUqWgMhwcgWV7^jyKWjVOu%p&VdrQj)*Da}+G|n_'
        't%=Y)7$M-`KzI=4B(sV{*KrsrU{o00jWT0hx4|tKEnrNz%#N5=upTM}9FnDJ5|cJNGCT!i1srOdXrspoI4lsZko%f0Yf5PbW)W-V'
        'nQj;_;Bebu%It{HUYck|^`rq4qM4Kt*YVg?CljV+A0+=$x!G)Q!jV(Xg^2==vQv+w3`@tNV3INweelZ(7W%`N<%alSvYdOgfIxt1'
        '<VJ_>?UbnekO#`YAl>7KsdVfz?VU!{vcI1eZO<l>@=%$6Ov(?_X?~{LN@UV{5~{X@i{*Zo(F4_N0+rK|)Z-@15Ku$5({6-kG8P24'
        'hRaFZ%p}(sJQCHVvrrq2W|B#%D2b*&W_0M;q!mj<rC43Bx*e$B1PvO@q#UHConvF~s6<*0niU=*zc!La%1GzRn9ZhU+v!+Fx^*tY'
        'Fh}tW**Uw9mz$2=$4JZO3OF{vCNZ?{I05qlYzmvorm=cfwS`r)@l4h^p9QzFT2{wob_+CUVvTYLDY7{XA<Y7gcOVM|w0I!n5VA;t'
        'EY{$J0>}~pAqTQlz%mbHEJ9ip$Z`!<6hKxA&>YAr0Vld4t;je@VVtbNDJr9tjE_jKEc7re;M4#sCUTpA(;V{Y0#<v{D@ODg3Ob^}'
        'nXdF!GKtdztVB65X?YPau{B%eh9BA`J1QV1b-W$l+vtWI)FB|AOQ6-(FF}2#=qEJjB;7VTAtP&KGk39487iZCk^)jrJ!t_MPd#PG'
        '>sEN60e98YhV_^MHi%)`mVhkcFuOGZ*1EGBgtT=Etw)3PL`kwP=9gO~MA;zVEQcZloZUx_=O~no8l0P_43=9%h;p8Q^Bu|s0yeoR'
        'Ly)prp<JlJmON#s+$tx^Rsk0|l#2yi;--`%<x+)mnFiY&io|8<9~(wqZx^t`@%nNBS9rW0hO8?U)>Rr@UD!L<unJ1-S^?KNiCr(?'
        '26ti=2)j{%-K4?I1li8ILVL6Arra4pY3&rS%Sr1N0k?Y68iAzS6w>V)+)<d;odGtIaCZr~+rix<;9d`IB;xK<aQAEQfD5-1aSzJP'
        '(ZqX5z{3vj5dn{Sc%zZ`n8JHpgC`1kPs+_o;yoqcX@~cWfM-3tO5{DK@SfM;g+ksp>;pTNEulpU7mAcz>^STukrxGg(-HZWfNy(5'
        'cB9C56p`<0@V!D$U!sMkU7qtM>|wYpzAWJT4)+HFUU9h7ZSEfAzN&Cv)8KW<ot5Xl$il1PhXLxKHw3)tbkL6k{Mgk&As@2dQdn<m'
        '@DpOZ%(@oM`{)&Xy($BU`l*0-9MsPQ{M?NyLe#qo>K7Wk*Ngi2)7Rb^ec`lhyFqP97vKD%|33M-SHLfwx_>3$1CO8Yqrk5ff!}EG'
        '+d@A-4A9z`Ojrh&*TH>c>3#vfbI`vR@R7&TeTe>pg8rihAG<8wi<bV0Z1_aLpB)?iBH*umIQlmQ^>+<ECB%N#6`mk7J8FX^Z^0rd'
        'c^?`{8+h+vtTUon%b1f*n|g;K;InMIE-k_v(X<>YZw)`ED!vf#52uQM3iy|&iqBE{--`5qH0~>s_pjbUc#(_-5XlF~Ey0UvpTPVe'
        'ij+`9#ZzUx6nS|CUM6=nKF}-j1&RzJgmjGi2{||?B7NgSC^EEHL~)OoyETT9#&G$L>`u&BUfqi9rsNew8lkWcrpU+wHZ~|9<z|m2'
        '_82$2m6{Y=l2;Pz5XGLc6ggC}r(VI2bK?&q{CL}*sj5AKA1<SG>dVzSYw#m*5Ee9n1ddb+nn;nOyajdA6e&8%EjpP*k9KORq?)$U'
        'S!d)IE15umfa6%RlSLxNr{G{Ld@5;9QwpC>k*b2iJE!E*sCK6@gVLz+*1iO#gGizNwtrIaT1U5zRO%Jo28zrq)UD0wHoA3Zk?w4d'
        'ZYLIm(sS_h2y*%uBF|NvK9(ZKd7YNyrOrIJ&V15Y;MiPex23)4(vO8^d0T04{ISHFobsAU;CQ9Hg%oKiEUzh7-Xgc`Vv;?fzw(y!'
        'mKP%OQl-3Q6lv|dyyb446{NG$DX&>APc{(UXySE@Yff#eh<>6{+es8Txv(}ZSKBFW%`j=6+GoqOA*D0lC#MnmbVY16Mb7BIPa<xy'
        'Gf7PM8o4TGWP7iXQ6k3_BMpjl^lfC^t;0ws;TU<EYGh}=F*=FZrPPz8NUHzFNV|11q|@DBJ<wYZC$g#3V^JjAcRg#|I%`R1ol{Sz'
        'r!jIDmmXq`XLIS|(!6xR;)Q&@Q|tzkJ4-26P~_~wV%O)2J;$BIMoQvbZ>Kb)^m%w;^*fJylJb_#;^#Y8*9%B!lT!O;id<N5bv-|)'
        'yv41&m6R`X%HKfcFPcBxethM)fO%X@NZayDNaRvQ_A-iW>)Z0}Zp|H}d3jEg&9(hz6WBzOTFtM>FSS<^_9{i_YKmOb_fmVUTj)9x'
        'y513*h&OKgufx~%x#ji-qTQ(QZ=%S}eJ{5=-TYm|zs1G>#Np%OM8{k3Tje_)fV{D0mev`;Z_8f?1izhnI<(iMb3*Vtoc%0aVFbTZ'
        '8GM&#@ZI^rS+-m7dvJFTd)Z{~3w|$Evrpc#ea^m+-xs?c>n!IngBvk?yyEBg$)Qs;em^!HE@*3wM4Vat0l6QwGCazvBUvkvH2o27'
        ';G>;b#4^%mhDX-a`uT&XW8bfua<CpZ_(S=L{+47$w^rG4KYth%<e#sGKZ2Y1s4Ci><&VmpP^rZqlb<U?zR-YB@n-%w28%;QoB0!R'
        'YinmN%O{KQ(cV*Z6rPsvrr^(D-IJUF&mtN9=FgEy&*MI(CvQ_tSv-FM!9}KNwr8tXgvtgkdvDw7&7*ui{9DoQQ{6GXhWu0AYJAkF'
        'Tag^Gr77rnxce1eL6m;#<9D$|@DBF?(qny%z0x_8Bl2D+p2jC?ElqV_U%jnuW<4e_Z!PQ}ASDJIFo6o+ynF)fJ(iKd?rCW{=6}(3'
        'Uh>C{l{7GiO~-z42)1460dpw#9p`$gKBK0kY0m$VfKA;vPrl8U|K#6B`L}C6W;AB|hdV0e7nKLhXqazWKBIgm9Q{>Os`I1L8#l*K'
        '`f~rr`)Yd|fQmN_o{$}|UdPipl8Ci5Rewz{f(?zevx2i{Hw1&T8tP}(*VWX{tf`;XSQo69-_+GK%nk-;&upy2EbHESeN{l_5uhxK'
        'Fxh;b>e(~AAan2kgLr4FC%_oj%(qAR`a9pe5VeWNx*#h@`wnpCcsXh#R=7BCDxCZY;GM2^O}T6O$zyz;-u2E^`(T)B?mlI1KlQ7J'
        'XS{2&R_Xha##s?6EuHSPS#ec<W|`;n^&<Q8vr79@&ZDnh${B87!AeT)Ib}579{XSVH{8q'
    ),
    4: (
        'c-pmCOKclO7+%-UIB8O+Q4*pmYNeK@l%%#}Crzr7HVH0tOGs0YDkQX8ug9Ks*SlsPd9;aCgjy0Y2N27F9uQpN!i^h8PCQx;NUcDs'
        '!~s=8YI{QMiGOAt$98P5iNz|8XT0-$|2N<KGqbacyw}HZp%Az0I>7aEqAp>18J)_ArZ`r-rle-Oxp%6ro4(JwA`W1~9w7lqQ?qi;'
        're%LY%OHgneX3YMHtCcUxhP1MVQK}N?3W8-4hcw=pq<Xz1q{v7CE5~Oxty%#vLe*5pS)O*C0#SLta-{<k=0RCR@X+0qAB6TDAY6B'
        '%H(J~p0G)n1r^M-BCz;qTllT0BU2Y;6=m#A%ObwCXdsA3!j1w%zK(4Ep+Um5p8{*j<a88dh$~2|R2>OP7_AnOZqiB+Dux+LK-nT4'
        '08x#srWc%!?=VD`L=ab4LNrEZPS=W#AEl7QZ%Cp7O8c||%8AsxF-11al8YpK2`Y0=<x~5_`;IE1fJ~7}wMq95j&+JeptVW6s=u+_'
        's0<A0C|hzS0AFCQO}yvTHE2OPwR9dyX6be)ZOICFY=DPU4{Z_Bil(foqGE3n?=`e$<0HhMRm7aZtYjrY*EG}KBHcPNG-Vm-f+T`b'
        'bmTQf0e&C=&dv<Vik4z#WQkJaIPsN=x9lZ|1niCwxZ{;o7QPE9sDRXxLL9ZyBq$U`9e%2^j}n#bn}<q`efT(jsZ<)pQ5?fFIJt=@'
        '@Gzd-u<_Iup2lbK4Ap^W;r^>M5>DaQo1+i4g&t}Lz0efAKtcl8VN#nY((9(h+gyu)vlQYdJv)}GFy&xWgOExKP^`2Xl5STXI`I~q'
        'aQ<E4SQt!v!ZM^94nQ^Nra9PLGj=q0V7*MUTt`YreJLAw9fzUhcY*sG!#knW<pS?&4Bro>ZWnlWL-?kZw%8x;ar3dKg|WvG7bTJU'
        'O1{GD{6i~kK&97BM{i3x*yl#m*PP~CN&}^SH=h31cm~~g23z8xBQ=Dtk`B7w7GR&yZ8ze(E%-(JX_$zHjgL0qN;q8i>v7GOx$fcY'
        'y8vmL0z6kg1s>q_3p0QJNUxvZhvpofCLE9d`T2hThbOHxVhKe;<qgrzsVA*u76~e{iFCUBxR~{<xmi!CS$p&9IbJ76j@~fgAwwv@'
        '1CyX4x}Bm`Bv~e!wp2sRBAPBD^gp0NGkCahk(&>DIIac7BQ89AbCP@~4YTJo$seTfb>6eb6e-~sSa!JNLmi6Bd?@vVP9E5dEaWnl'
        '`B*)F8PN%*X^W*RDlf=v89Rh8GkQ7Wmc9Au2IIGrz*n6F4mb~-u7q=Lg!wZi2k>E+W&d#fj?X)T>8$)DJO|GqNN3N%ZrZN=@7ms~'
        ')CS{0d~~|G3X@tK1;^dCM<CN?FO|w(I<B_ue7l-%)Hq+B48!;$@y}g4|Jtjr!|;*oFkEI+H_Q&53T&mO+x)c~&R@a=UpRNbIqGif'
        'c0}*Z1boqP03X4}7}cKl)njx|eDpcJ?(B76WgJhpIp`9_4BoJrDdmnE7F&e2vzulB9GqFkl*$&HAI%zA_%0Oh;|B(QLe)LP&)Lqz'
        'R$I91tvxNisb`RmGV{(6@dTU@11m_*VL148nVK%)F~Krr1)BB>I-*C{i~t87L(_%jI3Pw~M>n7$ob_QPX9z&16iqa*lzxT-rY8!g'
        'VX0`@e899X^BljA_ujU7FYmc+Z{@!SZ4lW6zYI`+eESE&v9|%RHyfbV<~@dCq^+?<J{VG#e)wY`!g2Jo*zh&W0C%+fnq`c>LJ&=%'
        '#Fm;pT=TYN_Z6d&zI*)p2Ox)|pT?1fxXGPJB|ik%cXBE{SEpvjU4zxRCK6P@%*uO@_Cbh!O$9`_nX-WNnu#<zuBq8Gdx^V!!IPS)'
        'AgbZk**{+hK$JjJ4c*);Rd2OUMN_lWZ3u8!Z_dEKbMObBoACMa991;%@n4?|!i~YEih8({WkuzIKi;aO@i2{a?~B~q?*0#R@tp)<'
        'oi~s!Lk3(I=~NTsOlo%QKY9^MPR38irlyjy*y&_qA~7C~PedmtVpGwnXfm0MjZY<F$??fas`BZbufG`q9V1Mohl(xhteh+db}6rY'
        '4_S+@oB#t=$XtYLG`(s`DwBAqisW7;$*!4G%Tt+%kt&|0UT*gUsD)Q%O?MUiw_dI$cWbegdC*gZ#Rs|CEUN`q4$p8E_}2<>{lr=1'
        '1A&*f)66?sMh0uXfAKiqt9{Nl;MlDT{Z8Fk!TbRS6J+h3w*La}RD-S'
    ),
}


class OmegaConfCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="inert-omegaconf-")
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "swint_mcc.tar"
        self.marker = self.root / "resolver-executed"

    def tearDown(self):
        self.temporary.cleanup()

    def write_fixture(self, protocol=2):
        self.checkpoint.write_bytes(zlib.decompress(base64.b85decode(_FIXTURES[protocol])))

    def rewrite_pickle(self, change):
        with zipfile.ZipFile(self.checkpoint) as archive:
            members = [(info, archive.read(info.filename)) for info in archive.infolist()]
        with zipfile.ZipFile(self.checkpoint, "w") as archive:
            for info, content in members:
                archive.writestr(info, change(content) if info.filename.endswith("/data.pkl") else content)

    @staticmethod
    def unicode(value):
        raw = value.encode("utf-8")
        return pickle.BINUNICODE + struct.pack("<I", len(raw)) + raw

    def append_root_value(self, expression):
        def change(content):
            self.assertEqual(content[-1:], pickle.STOP)
            return content[:-1] + self.unicode("attack") + expression + pickle.SETITEM + pickle.STOP
        self.rewrite_pickle(change)

    def load(self):
        return load_checkpoint_payload(self.checkpoint, allow_omegaconf_metadata=True)

    def test_real_official_conversion_parent_graph_is_plain_across_pickle_protocols(self):
        for protocol in (2, 3, 4):
            self.write_fixture(protocol)
            with self.subTest(protocol=protocol):
                loaded = self.load()
                config = loaded["config"]
                self.assertEqual(config["transform"]["train"]["RandomResizedCrop"]["scale"], [0.8, 1.1])
                self.assertEqual(config["optimizer"]["betas"], [0.9, 0.999])
                self.assertEqual(config["metadata"]["nested"], [[1, None], {"value": False}])
                torch.testing.assert_close(loaded["model"]["weight"], torch.arange(6, dtype=torch.float32).reshape(2, 3))
                self.assertEqual(loaded["model"]._metadata, OrderedDict([("", {"version": 1})]))
                self.assertEqual(_training_config(loaded), config)
                self.assertFalse(_preprocess(config)[0].do_hist_equalization)

    def test_omegaconf_is_not_imported_and_interpolations_never_execute(self):
        self.write_fixture()
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "omegaconf" or name.startswith("omegaconf."):
                raise AssertionError("OmegaConf code must not be imported")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            loaded = self.load()
        self.assertEqual(loaded["config"]["metadata"]["literal"], ["${test_must_never_execute:unsafe}"])
        self.assertFalse(self.marker.exists())

    def test_explicit_opt_in_is_required_and_numeric_allowlist_is_unchanged(self):
        self.write_fixture()
        with mock.patch("foundation_sft.models.common._load_numeric_checkpoint", side_effect=AssertionError("No numeric fallback")):
            with self.assertRaises(pickle.UnpicklingError):
                load_checkpoint_payload(self.checkpoint)

    def test_compatibility_uses_only_the_local_restricted_unpickler(self):
        self.write_fixture()
        real_load = torch.load
        with mock.patch("foundation_sft.models.common.torch.load", wraps=real_load) as load:
            self.load()
        self.assertEqual(load.call_count, 1)
        self.assertFalse(load.call_args_list[-1].kwargs["weights_only"])
        self.assertEqual(load.call_args_list[-1].kwargs["pickle_module"].Unpickler.__name__, "_OmegaConfCheckpointUnpickler")

    def test_malformed_build_cannot_add_metadata_methods_or_fields(self):
        self.write_fixture()

        def change(content):
            original = self.unicode("_content")
            self.assertIn(original, content)
            return content.replace(original, self.unicode("_danger_"), 1)

        self.rewrite_pickle(change)
        with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported OmegaConf checkpoint metadata state"):
            self.load()

    def test_newobj_accepts_only_known_inert_classes_with_no_constructor_arguments(self):
        self.write_fixture()

        def change(content):
            operations = list(pickletools.genops(content))
            for index, (operation, argument, position) in enumerate(operations):
                if operation.name == "NEWOBJ":
                    previous, _, start = operations[index - 1]
                    self.assertEqual(previous.name, "EMPTY_TUPLE")
                    return content[:start] + pickle.MARK + pickle.BININT1 + b"\x01" + pickle.TUPLE + content[position:]
            self.fail("Fixture must contain real OmegaConf NEWOBJ nodes")

        self.rewrite_pickle(change)
        with self.assertRaisesRegex(pickle.UnpicklingError, "metadata constructor"):
            self.load()
        self.write_fixture()
        self.append_root_value(pickle.GLOBAL + b"builtins\ndict\n" + pickle.EMPTY_TUPLE + pickle.NEWOBJ)
        with self.assertRaisesRegex(pickle.UnpicklingError, "metadata constructor"):
            self.load()

    def test_metadata_types_cannot_be_called_through_reduce(self):
        self.write_fixture()
        self.append_root_value(pickle.GLOBAL + b"omegaconf.listconfig\nListConfig\n" + pickle.EMPTY_TUPLE + pickle.REDUCE)
        with self.assertRaisesRegex(pickle.UnpicklingError, "Unrecognized checkpoint reconstruction function"):
            self.load()

    def test_unknown_application_globals_and_eval_remain_rejected(self):
        self.write_fixture()
        self.append_root_value(pickle.GLOBAL + b"argparse\nNamespace\n" + pickle.EMPTY_TUPLE + pickle.NEWOBJ)
        with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported checkpoint global:.*Namespace"):
            self.load()
        self.write_fixture()
        expression = f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')"
        self.append_root_value(pickle.GLOBAL + b"builtins\neval\n" + self.unicode(expression) + pickle.TUPLE1 + pickle.REDUCE)
        with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported checkpoint global:.*eval"):
            self.load()
        self.assertFalse(self.marker.exists())

    def test_resolver_cache_rejects_factories_other_than_inert_empty_dict(self):
        self.write_fixture()
        self.append_root_value(pickle.GLOBAL + b"collections\ndefaultdict\n" + pickle.GLOBAL + b"builtins\nlist\n" + pickle.TUPLE1 + pickle.REDUCE)
        with self.assertRaisesRegex(pickle.UnpicklingError, "resolver-cache factory"):
            self.load()

    def test_config_value_cycles_and_type_tokens_are_rejected(self):
        self.write_fixture()
        self.append_root_value(pickle.GLOBAL + b"typing\nAny\n")
        with self.assertRaisesRegex(pickle.UnpicklingError, "type metadata cannot be a configuration value"):
            self.load()
        # A normal plain value cycle is malformed configuration, unlike the
        # allowed OmegaConf parent pointers discarded during conversion.
        from foundation_sft.models.common import _plain_omegaconf_metadata
        value = []
        value.append(value)
        with self.assertRaisesRegex(pickle.UnpicklingError, "Cyclic checkpoint configuration"):
            _plain_omegaconf_metadata({"config": value})

    def test_torch_legacy_tar_reader_is_never_used_for_metadata_compatibility(self):
        torch.save({"model": {"weight": torch.zeros(2)}, "metric": np.float64(0.8)}, self.checkpoint,
                   _use_new_zipfile_serialization=False)
        with self.assertRaisesRegex(pickle.UnpicklingError, "requires a Torch ZIP checkpoint"):
            self.load()

    def test_nonboolean_opt_in_is_rejected(self):
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            load_checkpoint_payload(self.checkpoint, allow_omegaconf_metadata="yes")


if __name__ == "__main__":
    unittest.main()
