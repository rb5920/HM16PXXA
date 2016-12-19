#include "CUFeature.h"

void  CUFeature::init()
{
    SUMSIZE64=0;
    SUMVSIZE64=0;
    AVGSIZE64=0;
    VARSIZE64=0;
    //QP=0;
    for(int d1 = 0; d1 < 4; d1++)//Elevate SUM/SUMV/AVG/VAR64 by adding SUM/SUMV4
    {
        for(int d2 = 0; d2 < 4; d2++)
        {
            for(int d3 = 0; d3 < 4; d3++)
            {
                for(int step = 0; step < 4; step++)
                {
                    SUMSIZE4[step+d3*4+d2*16+d1*64]=0;
                    SUMVSIZE4[step+d3*4+d2*16+d1*64]=0;
                    AVGSIZE4[step+d3*4+d2*16+d1*64]=0;
                    VARSIZE4[step+d3*4+d2*16+d1*64]=0;
                }

                SUMSIZE8[d1][d2][d3]=0;
                SUMVSIZE8[d1][d2][d3]=0;
                AVGSIZE8[d1][d2][d3]=0;
                VARSIZE8[d1][d2][d3]=0;
            }

            SUMSIZE16[d1][d2]=0;
            SUMVSIZE16[d1][d2]=0;
            AVGSIZE16[d1][d2]=0;
            VARSIZE16[d1][d2]=0;
        }

        SUMSIZE32[d1]=0;
        SUMVSIZE32[d1]=0;
        AVGSIZE32[d1]=0;
        VARSIZE32[d1]=0;
    }
}
/** 8x8 forward transform implemented using partial butterfly structure (1D)
*  \param src   input data (residual)
*  \param dst   output data (transform coefficients)
*  \param shift specifies right shift after 1D transform
*  \param line
*/
Void CUFeature::MYpartialButterfly8(TCoeff *src, TCoeff *dst, Int shift, Int line)
{
	Int j, k;
	TCoeff E[4], O[4];
	TCoeff EE[2], EO[2];
	TCoeff add = (shift > 0) ? (1 << (shift - 1)) : 0;

	for (j = 0; j<line; j++)
	{
		/* E and O*/
		for (k = 0; k<4; k++)
		{
			E[k] = src[k] + src[7 - k];
			O[k] = src[k] - src[7 - k];
		}
		/* EE and EO */
		EE[0] = E[0] + E[3];
		EO[0] = E[0] - E[3];
		EE[1] = E[1] + E[2];
		EO[1] = E[1] - E[2];

		dst[0] = (g_aiT8[TRANSFORM_FORWARD][0][0] * EE[0] + g_aiT8[TRANSFORM_FORWARD][0][1] * EE[1] + add) >> shift;
		dst[4 * line] = (g_aiT8[TRANSFORM_FORWARD][4][0] * EE[0] + g_aiT8[TRANSFORM_FORWARD][4][1] * EE[1] + add) >> shift;
		dst[2 * line] = (g_aiT8[TRANSFORM_FORWARD][2][0] * EO[0] + g_aiT8[TRANSFORM_FORWARD][2][1] * EO[1] + add) >> shift;
		dst[6 * line] = (g_aiT8[TRANSFORM_FORWARD][6][0] * EO[0] + g_aiT8[TRANSFORM_FORWARD][6][1] * EO[1] + add) >> shift;

		dst[line] = (g_aiT8[TRANSFORM_FORWARD][1][0] * O[0] + g_aiT8[TRANSFORM_FORWARD][1][1] * O[1] + g_aiT8[TRANSFORM_FORWARD][1][2] * O[2] + g_aiT8[TRANSFORM_FORWARD][1][3] * O[3] + add) >> shift;
		dst[3 * line] = (g_aiT8[TRANSFORM_FORWARD][3][0] * O[0] + g_aiT8[TRANSFORM_FORWARD][3][1] * O[1] + g_aiT8[TRANSFORM_FORWARD][3][2] * O[2] + g_aiT8[TRANSFORM_FORWARD][3][3] * O[3] + add) >> shift;
		dst[5 * line] = (g_aiT8[TRANSFORM_FORWARD][5][0] * O[0] + g_aiT8[TRANSFORM_FORWARD][5][1] * O[1] + g_aiT8[TRANSFORM_FORWARD][5][2] * O[2] + g_aiT8[TRANSFORM_FORWARD][5][3] * O[3] + add) >> shift;
		dst[7 * line] = (g_aiT8[TRANSFORM_FORWARD][7][0] * O[0] + g_aiT8[TRANSFORM_FORWARD][7][1] * O[1] + g_aiT8[TRANSFORM_FORWARD][7][2] * O[2] + g_aiT8[TRANSFORM_FORWARD][7][3] * O[3] + add) >> shift;

		src += 8;
		dst++;
	}
}
/** 16x16 forward transform implemented using partial butterfly structure (1D)
*  \param src   input data (residual)
*  \param dst   output data (transform coefficients)
*  \param shift specifies right shift after 1D transform
*  \param line
*/
Void CUFeature::MYpartialButterfly16(TCoeff *src, TCoeff *dst, Int shift, Int line)
{
	Int j, k;
	TCoeff E[8], O[8];
	TCoeff EE[4], EO[4];
	TCoeff EEE[2], EEO[2];
	TCoeff add = (shift > 0) ? (1 << (shift - 1)) : 0;

	for (j = 0; j<line; j++)
	{
		/* E and O*/
		for (k = 0; k<8; k++)
		{
			E[k] = src[k] + src[15 - k];
			O[k] = src[k] - src[15 - k];
		}
		/* EE and EO */
		for (k = 0; k<4; k++)
		{
			EE[k] = E[k] + E[7 - k];
			EO[k] = E[k] - E[7 - k];
		}
		/* EEE and EEO */
		EEE[0] = EE[0] + EE[3];
		EEO[0] = EE[0] - EE[3];
		EEE[1] = EE[1] + EE[2];
		EEO[1] = EE[1] - EE[2];

		dst[0] = (g_aiT16[TRANSFORM_FORWARD][0][0] * EEE[0] + g_aiT16[TRANSFORM_FORWARD][0][1] * EEE[1] + add) >> shift;
		dst[8 * line] = (g_aiT16[TRANSFORM_FORWARD][8][0] * EEE[0] + g_aiT16[TRANSFORM_FORWARD][8][1] * EEE[1] + add) >> shift;
		dst[4 * line] = (g_aiT16[TRANSFORM_FORWARD][4][0] * EEO[0] + g_aiT16[TRANSFORM_FORWARD][4][1] * EEO[1] + add) >> shift;
		dst[12 * line] = (g_aiT16[TRANSFORM_FORWARD][12][0] * EEO[0] + g_aiT16[TRANSFORM_FORWARD][12][1] * EEO[1] + add) >> shift;

		for (k = 2; k<16; k += 4)
		{
			dst[k*line] = (g_aiT16[TRANSFORM_FORWARD][k][0] * EO[0] + g_aiT16[TRANSFORM_FORWARD][k][1] * EO[1] +
				g_aiT16[TRANSFORM_FORWARD][k][2] * EO[2] + g_aiT16[TRANSFORM_FORWARD][k][3] * EO[3] + add) >> shift;
		}

		for (k = 1; k<16; k += 2)
		{
			dst[k*line] = (g_aiT16[TRANSFORM_FORWARD][k][0] * O[0] + g_aiT16[TRANSFORM_FORWARD][k][1] * O[1] +
				g_aiT16[TRANSFORM_FORWARD][k][2] * O[2] + g_aiT16[TRANSFORM_FORWARD][k][3] * O[3] +
				g_aiT16[TRANSFORM_FORWARD][k][4] * O[4] + g_aiT16[TRANSFORM_FORWARD][k][5] * O[5] +
				g_aiT16[TRANSFORM_FORWARD][k][6] * O[6] + g_aiT16[TRANSFORM_FORWARD][k][7] * O[7] + add) >> shift;
		}

		src += 16;
		dst++;

	}
}
/** 32x32 forward transform implemented using partial butterfly structure (1D)
*  \param src   input data (residual)
*  \param dst   output data (transform coefficients)
*  \param shift specifies right shift after 1D transform
*  \param line
*/
Void CUFeature::MYpartialButterfly32(TCoeff *src, TCoeff *dst, Int shift, Int line)
{
	Int j, k;
	TCoeff E[16], O[16];
	TCoeff EE[8], EO[8];
	TCoeff EEE[4], EEO[4];
	TCoeff EEEE[2], EEEO[2];
	TCoeff add = (shift > 0) ? (1 << (shift - 1)) : 0;

	for (j = 0; j<line; j++)
	{
		/* E and O*/
		for (k = 0; k<16; k++)
		{
			E[k] = src[k] + src[31 - k];
			O[k] = src[k] - src[31 - k];
		}
		/* EE and EO */
		for (k = 0; k<8; k++)
		{
			EE[k] = E[k] + E[15 - k];
			EO[k] = E[k] - E[15 - k];
		}
		/* EEE and EEO */
		for (k = 0; k<4; k++)
		{
			EEE[k] = EE[k] + EE[7 - k];
			EEO[k] = EE[k] - EE[7 - k];
		}
		/* EEEE and EEEO */
		EEEE[0] = EEE[0] + EEE[3];
		EEEO[0] = EEE[0] - EEE[3];
		EEEE[1] = EEE[1] + EEE[2];
		EEEO[1] = EEE[1] - EEE[2];

		dst[0] = (g_aiT32[TRANSFORM_FORWARD][0][0] * EEEE[0] + g_aiT32[TRANSFORM_FORWARD][0][1] * EEEE[1] + add) >> shift;
		dst[16 * line] = (g_aiT32[TRANSFORM_FORWARD][16][0] * EEEE[0] + g_aiT32[TRANSFORM_FORWARD][16][1] * EEEE[1] + add) >> shift;
		dst[8 * line] = (g_aiT32[TRANSFORM_FORWARD][8][0] * EEEO[0] + g_aiT32[TRANSFORM_FORWARD][8][1] * EEEO[1] + add) >> shift;
		dst[24 * line] = (g_aiT32[TRANSFORM_FORWARD][24][0] * EEEO[0] + g_aiT32[TRANSFORM_FORWARD][24][1] * EEEO[1] + add) >> shift;
		for (k = 4; k<32; k += 8)
		{
			dst[k*line] = (g_aiT32[TRANSFORM_FORWARD][k][0] * EEO[0] + g_aiT32[TRANSFORM_FORWARD][k][1] * EEO[1] +
				g_aiT32[TRANSFORM_FORWARD][k][2] * EEO[2] + g_aiT32[TRANSFORM_FORWARD][k][3] * EEO[3] + add) >> shift;
		}
		for (k = 2; k<32; k += 4)
		{
			dst[k*line] = (g_aiT32[TRANSFORM_FORWARD][k][0] * EO[0] + g_aiT32[TRANSFORM_FORWARD][k][1] * EO[1] +
				g_aiT32[TRANSFORM_FORWARD][k][2] * EO[2] + g_aiT32[TRANSFORM_FORWARD][k][3] * EO[3] +
				g_aiT32[TRANSFORM_FORWARD][k][4] * EO[4] + g_aiT32[TRANSFORM_FORWARD][k][5] * EO[5] +
				g_aiT32[TRANSFORM_FORWARD][k][6] * EO[6] + g_aiT32[TRANSFORM_FORWARD][k][7] * EO[7] + add) >> shift;
		}
		for (k = 1; k<32; k += 2)
		{
			dst[k*line] = (g_aiT32[TRANSFORM_FORWARD][k][0] * O[0] + g_aiT32[TRANSFORM_FORWARD][k][1] * O[1] +
				g_aiT32[TRANSFORM_FORWARD][k][2] * O[2] + g_aiT32[TRANSFORM_FORWARD][k][3] * O[3] +
				g_aiT32[TRANSFORM_FORWARD][k][4] * O[4] + g_aiT32[TRANSFORM_FORWARD][k][5] * O[5] +
				g_aiT32[TRANSFORM_FORWARD][k][6] * O[6] + g_aiT32[TRANSFORM_FORWARD][k][7] * O[7] +
				g_aiT32[TRANSFORM_FORWARD][k][8] * O[8] + g_aiT32[TRANSFORM_FORWARD][k][9] * O[9] +
				g_aiT32[TRANSFORM_FORWARD][k][10] * O[10] + g_aiT32[TRANSFORM_FORWARD][k][11] * O[11] +
				g_aiT32[TRANSFORM_FORWARD][k][12] * O[12] + g_aiT32[TRANSFORM_FORWARD][k][13] * O[13] +
				g_aiT32[TRANSFORM_FORWARD][k][14] * O[14] + g_aiT32[TRANSFORM_FORWARD][k][15] * O[15] + add) >> shift;
		}

		src += 32;
		dst++;
	}
}
/** 4x4 forward transform implemented using partial butterfly structure (1D)
*  \param src   input data (residual)
*  \param dst   output data (transform coefficients)
*  \param shift specifies right shift after 1D transform
*  \param line
*/
Void CUFeature::MYpartialButterfly4(TCoeff *src, TCoeff *dst, Int shift, Int line)
{
	Int j;
	TCoeff E[2], O[2];
	TCoeff add = (shift > 0) ? (1 << (shift - 1)) : 0;

	for (j = 0; j<line; j++)
	{
		/* E and O */
		E[0] = src[0] + src[3];
		O[0] = src[0] - src[3];
		E[1] = src[1] + src[2];
		O[1] = src[1] - src[2];

		dst[0] = (g_aiT4[TRANSFORM_FORWARD][0][0] * E[0] + g_aiT4[TRANSFORM_FORWARD][0][1] * E[1] + add) >> shift;
		dst[2 * line] = (g_aiT4[TRANSFORM_FORWARD][2][0] * E[0] + g_aiT4[TRANSFORM_FORWARD][2][1] * E[1] + add) >> shift;
		dst[line] = (g_aiT4[TRANSFORM_FORWARD][1][0] * O[0] + g_aiT4[TRANSFORM_FORWARD][1][1] * O[1] + add) >> shift;
		dst[3 * line] = (g_aiT4[TRANSFORM_FORWARD][3][0] * O[0] + g_aiT4[TRANSFORM_FORWARD][3][1] * O[1] + add) >> shift;

		src += 4;
		dst++;
	}
}
/** MxN forward transform (2D)
*  \param bitDepth              [in]  bit depth
*  \param block                 [in]  residual block
*  \param coeff                 [out] transform coefficients
*  \param iWidth                [in]  width of transform
*  \param iHeight               [in]  height of transform
*  \param useDST                [in]
*  \param maxLog2TrDynamicRange [in]

*/
Void CUFeature::MYxTrMxN(TCoeff *block, TCoeff *coeff, Int iWidth, Int iHeight)
{

	TCoeff tmp[MAX_TU_SIZE * MAX_TU_SIZE];

	switch (iWidth)
	{
	case 4:		MYpartialButterfly4(block, tmp, 0, iHeight);	break;
	case 8:     MYpartialButterfly8(block, tmp, 0, iHeight);  break;
	case 16:    MYpartialButterfly16(block, tmp, 0, iHeight);  break;
	case 32:    MYpartialButterfly32(block, tmp, 0, iHeight);  break;
	default:
		assert(0); exit(1); break;
	}

	switch (iHeight)
	{
	case 4:		MYpartialButterfly4(tmp, coeff, 0, iWidth);	break;
	case 8:     MYpartialButterfly8(tmp, coeff, 0, iWidth);    break;
	case 16:    MYpartialButterfly16(tmp, coeff, 0, iWidth);    break;
	case 32:    MYpartialButterfly32(tmp, coeff, 0, iWidth);    break;
	default:
		assert(0); exit(1); break;
	}
}
Double CUFeature::Average(Double* Data, Double data_num)
{
	Double SUM=0;
	for(int count = 0;count < data_num;count++)
	{
		SUM += Data[count];
	}
	return SUM/data_num;
}

void CUFeature::Variance(Double* Data, Double data_num, Double* Output1, Double* Output2, int mode)
{	
    Double SUMv=0;
    if(mode==VAR_NORMAL_MODE)
    {
        for(int count = 0;count < data_num;count++)
        {
            SUMv += Data[count] * Data[count];
        }
        *Output2 = Average(Data, data_num);
        *Output1 = (SUMv / data_num) - ((*Output2) * (*Output2));
    }
    else
    {
        *Output2 = Data[0] / data_num;
        *Output1 = (Data[1] / data_num) - ((*Output2) * (*Output2));
    }
}

void CUFeature::ElevateFeature(TComDataCU* pcCU, UInt uiAbsPartIdx, UInt uiDepth)
{
    Pel *ptrSrc_5920 = pcCU->getPic()->getPicYuvOrg()->getAddr(COMPONENT_Y, pcCU->getCtuRsAddr(), uiAbsPartIdx);
    int picwid = (int)(pcCU->getPic()->getPicYuvOrg()->getWidth(COMPONENT_Y)) + 160;
    //QP = pcCU->getQP(0);
    int LDepth1[5],LDepth2[5],LDepth3[5],INDEX;
    Double VARINPUT[2];
    Double AVGofROW64[64]={0};
    Double AVGofCOL64[64]={0};
    Double SUMAVGofROW64=0;
    Double SUMVAVGofROW64=0;
    Double AVGAVGofROW64=0;
    Double SUMAVGofCOL64=0;
    Double SUMVAVGofCOL64=0;
    Double AVGAVGofCOL64=0;
    TCoeff DCTIN81[1024];
    TCoeff DCTIN82[1024];
    TCoeff DCTIN83[1024];
    TCoeff DCTIN84[1024];
    TCoeff DCTOUT81[1024];
    TCoeff DCTOUT82[1024];
    TCoeff DCTOUT83[1024];
    TCoeff DCTOUT84[1024];
    TCoeff DCTIN16x16[16][256];
    TCoeff DCTOUT16x16[16][256];
	int DCTCOUNT1=0;
	int DCTCOUNT2=0;
	int DCTCOUNT3=0;
	int DCTCOUNT4=0;
	int DCTCOUNT_s1=0;
	int DCTCOUNT_s2=0;
	int DCTCOUNT_s3=0;
	int DCTCOUNT_s4=0;
	int DCTCOUNT_s5=0;
	int DCTCOUNT_s6=0;
	int DCTCOUNT_s7=0;
	int DCTCOUNT_s8=0;
	int DCTCOUNT_s9=0;
	int DCTCOUNT_s10=0;
	int DCTCOUNT_s11=0;
	int DCTCOUNT_s12=0;
	int DCTCOUNT_s13=0;
	int DCTCOUNT_s14=0;
	int DCTCOUNT_s15=0;
	int DCTCOUNT_s16=0;
	double DCTSUM81=0;
	double DCTSUM82=0;
	double DCTSUM83=0;
	double DCTSUM84=0;
    for(int row = 0; row < 64; row++)//Elevate SUM/SUMV4
    {
        for(int col = 0; col < 64; col++)//Depth0
        {
			AVGofCOL64[col] = AVGofCOL64[col] + ptrSrc_5920[col + row * picwid];
			AVGofROW64[row] = AVGofROW64[row] + ptrSrc_5920[col + row * picwid];
            if(row < 32 && col < 32)//Depth1
            {
                DCTIN81[DCTCOUNT1] = ptrSrc_5920[col + row * picwid];
                DCTCOUNT1++;
                LDepth1[0]=0;
                LDepth1[1]=0;
                LDepth1[2]=0;
                LDepth1[3]=LDepth1[1] * 32 + LDepth1[2] * 32;
                LDepth1[4]=LDepth1[0] * 32 + LDepth1[2] * 32;
                if(row < (16 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s1];
                    DCTCOUNT_s1++;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row < (16 + LDepth1[3]) && col > (15 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=1;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s2];
                    DCTCOUNT_s2++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row > (15 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=1;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s3];
                    DCTCOUNT_s3++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=1;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s4];
                    DCTCOUNT_s4++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
            }
            else if(row < 32 && col > 31)//Depth1
            {
                DCTIN82[DCTCOUNT2] = ptrSrc_5920[col + row * picwid];
                DCTCOUNT2++;
                LDepth1[0]=1;
                LDepth1[1]=0;
                LDepth1[2]=0;
                LDepth1[3]=LDepth1[1] * 32 + LDepth1[2] * 32;
                LDepth1[4]=LDepth1[0] * 32 + LDepth1[2] * 32;
                if(row < (16 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s5];
                    DCTCOUNT_s5++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row < (16 + LDepth1[3]) && col > (15 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=1;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s6];
                    DCTCOUNT_s6++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row > (15 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=1;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s7];
                    DCTCOUNT_s7++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=1;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s8];
                    DCTCOUNT_s8++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
            }
            else if(row > 31 && col < 32)//Depth1
            {
                DCTIN83[DCTCOUNT3] = ptrSrc_5920[col + row * picwid];
                DCTCOUNT3++;
                LDepth1[0]=0;
                LDepth1[1]=1;
                LDepth1[2]=0;
                LDepth1[3]=LDepth1[1] * 32 + LDepth1[2] * 32;
                LDepth1[4]=LDepth1[0] * 32 + LDepth1[2] * 32;
                if(row < (16 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s9];
                    DCTCOUNT_s9++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row < (16 + LDepth1[3]) && col > (15 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=1;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s10];
                    DCTCOUNT_s10++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row > (15 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=1;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s11];
                    DCTCOUNT_s11++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=1;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s12];
                    DCTCOUNT_s12++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
            }
            else//Depth1
            {
                DCTIN84[DCTCOUNT4] = ptrSrc_5920[col + row * picwid];
                DCTCOUNT4++;
                LDepth1[0]=0;
                LDepth1[1]=0;
                LDepth1[2]=1;
                LDepth1[3]=LDepth1[1] * 32 + LDepth1[2] * 32;
                LDepth1[4]=LDepth1[0] * 32 + LDepth1[2] * 32;
                if(row < (16 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s13];
                    DCTCOUNT_s13++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row < (16 + LDepth1[3]) && col > (15 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=1;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s14];
                    DCTCOUNT_s14++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row > (15 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=1;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s15];
                    DCTCOUNT_s15++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=1;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s16];
                    DCTCOUNT_s16++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
            }
        }
    }
	MYxTrMxN(DCTIN81, DCTOUT81, 32, 32);
	MYxTrMxN(DCTIN82, DCTOUT82, 32, 32);
	MYxTrMxN(DCTIN83, DCTOUT83, 32, 32);
	MYxTrMxN(DCTIN84, DCTOUT84, 32, 32);
	DCTOUT81[0]=DCTOUT81[0]>>17;
	DCTOUT82[0]=DCTOUT82[0]>>17;
	DCTOUT83[0]=DCTOUT83[0]>>17;
	DCTOUT84[0]=DCTOUT84[0]>>17;
	for(int stepi=1;stepi<1024;stepi++)
	{
		DCTOUT81[stepi]=abs(DCTOUT81[stepi]>>17);
		DCTSUM81=DCTSUM81+(double)DCTOUT81[stepi];
		DCTOUT82[stepi]=abs(DCTOUT82[stepi]>>17);
		DCTSUM82=DCTSUM82+(double)DCTOUT82[stepi];
		DCTOUT83[stepi]=abs(DCTOUT83[stepi]>>17);
		DCTSUM83=DCTSUM83+(double)DCTOUT83[stepi];
		DCTOUT84[stepi]=abs(DCTOUT84[stepi]>>17);
		DCTSUM84=DCTSUM84+(double)DCTOUT84[stepi];
	}
	DC_AVG =(double)( DCTOUT81[0] + DCTOUT82[0] + DCTOUT83[0] + DCTOUT84[0] )/ 4.0;
	AC_AVG =(DCTSUM81+DCTSUM82+DCTSUM83+DCTSUM84)/4.0;
    for(int step = 0; step < 64; step++)
    {
        AVGofCOL64[step] = AVGofCOL64[step] / 64.0;
        SUMAVGofCOL64 += AVGofCOL64[step];
        SUMVAVGofCOL64 += AVGofCOL64[step] * AVGofCOL64[step];
        AVGofROW64[step] = AVGofROW64[step] / 64.0;
        SUMAVGofROW64 += AVGofROW64[step];
        SUMVAVGofROW64 += AVGofROW64[step] * AVGofROW64[step];
    }
    AVGAVGofCOL64 = SUMAVGofCOL64 / 64.0;
    VARAVGofCOL64 = SUMVAVGofCOL64 / 64.0 - AVGAVGofCOL64 * AVGAVGofCOL64;
    AVGAVGofROW64 = SUMAVGofROW64 / 64.0;
    VARAVGofROW64 = SUMVAVGofROW64 / 64.0 - AVGAVGofROW64 * AVGAVGofROW64;
    for(int d1 = 0; d1 < 4; d1++)//Elevate SUM/SUMV/AVG/VAR64 by adding SUM/SUMV4
    {
        for(int d2 = 0; d2 < 4; d2++)
        {
            for(int d3 = 0; d3 < 4; d3++)
            {
                for(int step = 0; step < 4; step++)
                {
                    SUMSIZE8[d1][d2][d3] += SUMSIZE4[step+d3*4+d2*16+d1*64];
                    SUMVSIZE8[d1][d2][d3] += SUMVSIZE4[step+d3*4+d2*16+d1*64];
                    VARINPUT[0] = SUMSIZE4[step+d3*4+d2*16+d1*64];
                    VARINPUT[1] = SUMVSIZE4[step+d3*4+d2*16+d1*64];
                    Variance(VARINPUT, 16.0, &VARSIZE4[step+d3*4+d2*16+d1*64], &AVGSIZE4[step+d3*4+d2*16+d1*64], VAR_SUM_MODE);
                }

                SUMSIZE16[d1][d2] += SUMSIZE8[d1][d2][d3];
                SUMVSIZE16[d1][d2] += SUMVSIZE8[d1][d2][d3];
                VARINPUT[0] = SUMSIZE8[d1][d2][d3];
                VARINPUT[1] = SUMVSIZE8[d1][d2][d3];
                Variance(VARINPUT, 64.0, &VARSIZE8[d1][d2][d3], &AVGSIZE8[d1][d2][d3], VAR_SUM_MODE);
            }

            SUMSIZE32[d1] += SUMSIZE16[d1][d2];
            SUMVSIZE32[d1] += SUMVSIZE16[d1][d2];
            VARINPUT[0] = SUMSIZE16[d1][d2];
            VARINPUT[1] = SUMVSIZE16[d1][d2];
            Variance(VARINPUT, 256.0, &VARSIZE16[d1][d2], &AVGSIZE16[d1][d2], VAR_SUM_MODE);
        }

        SUMSIZE64 += SUMSIZE32[d1];
        SUMVSIZE64 += SUMVSIZE32[d1];
        VARINPUT[0] = SUMSIZE32[d1];
        VARINPUT[1] = SUMVSIZE32[d1];
        Variance(VARINPUT, 1024.0, &VARSIZE32[d1], &AVGSIZE32[d1], VAR_SUM_MODE);
    }
    VARINPUT[0] = SUMSIZE64;
    VARINPUT[1] = SUMVSIZE64;
    Variance(VARINPUT, 4096.0, &VARSIZE64, &AVGSIZE64, VAR_SUM_MODE);
}

void CUFeature::PreElevateFeature(TComPic* pcPic, UInt CtuIndex)
{
    Pel *ptrSrc_5920 = pcPic->getPicYuvOrg()->getAddr(COMPONENT_Y, CtuIndex, 0);
    int picwid = (int)(pcPic->getPicYuvOrg()->getWidth(COMPONENT_Y)) + 160;
    //const UInt uiMaxCuWidth  = pcPic->getPicSym()->getSPS()->getMaxCUWidth();
    //QP = 22;
    int LDepth1[5],LDepth2[5],LDepth3[5],INDEX;
    Double VARINPUT[2];
    Double AVGofROW64[64]={0};
    Double AVGofCOL64[64]={0};
    Double SUMAVGofROW64=0;
    Double SUMVAVGofROW64=0;
    Double AVGAVGofROW64=0;
    Double SUMAVGofCOL64=0;
    Double SUMVAVGofCOL64=0;
    Double AVGAVGofCOL64=0;
    TCoeff DCTIN81[1024];
    TCoeff DCTIN82[1024];
    TCoeff DCTIN83[1024];
    TCoeff DCTIN84[1024];
    TCoeff DCTIN16x16[16][256];
    TCoeff DCTOUT16x16[16][256];
    TCoeff DCTOUT81[1024];
    TCoeff DCTOUT82[1024];
    TCoeff DCTOUT83[1024];
    TCoeff DCTOUT84[1024];
	int DCTCOUNT1=0;
	int DCTCOUNT2=0;
	int DCTCOUNT3=0;
	int DCTCOUNT4=0;
	int DCTCOUNT_s1=0;
	int DCTCOUNT_s2=0;
	int DCTCOUNT_s3=0;
	int DCTCOUNT_s4=0;
	int DCTCOUNT_s5=0;
	int DCTCOUNT_s6=0;
	int DCTCOUNT_s7=0;
	int DCTCOUNT_s8=0;
	int DCTCOUNT_s9=0;
	int DCTCOUNT_s10=0;
	int DCTCOUNT_s11=0;
	int DCTCOUNT_s12=0;
	int DCTCOUNT_s13=0;
	int DCTCOUNT_s14=0;
	int DCTCOUNT_s15=0;
	int DCTCOUNT_s16=0;
	double DCTSUM81=0;
	double DCTSUM82=0;
	double DCTSUM83=0;
	double DCTSUM84=0;
    for(int row = 0; row < 64; row++)//Elevate SUM/SUMV4
    {
        for(int col = 0; col < 64; col++)//Depth0
        {
			AVGofCOL64[col] = AVGofCOL64[col] + ptrSrc_5920[col + row * picwid];
			AVGofROW64[row] = AVGofROW64[row] + ptrSrc_5920[col + row * picwid];
            if(row < 32 && col < 32)//Depth1
            {
                DCTIN81[DCTCOUNT1] = ptrSrc_5920[col + row * picwid];
                DCTCOUNT1++;
                LDepth1[0]=0;
                LDepth1[1]=0;
                LDepth1[2]=0;
                LDepth1[3]=LDepth1[1] * 32 + LDepth1[2] * 32;
                LDepth1[4]=LDepth1[0] * 32 + LDepth1[2] * 32;
                if(row < (16 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s1];
                    DCTCOUNT_s1++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row < (16 + LDepth1[3]) && col > (15 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=1;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s2];
                    DCTCOUNT_s2++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row > (15 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=1;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s3];
                    DCTCOUNT_s3++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=1;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s4];
                    DCTCOUNT_s4++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
            }
            else if(row < 32 && col > 31)//Depth1
            {
                DCTIN82[DCTCOUNT2] = ptrSrc_5920[col + row * picwid];
                DCTCOUNT2++;
                LDepth1[0]=1;
                LDepth1[1]=0;
                LDepth1[2]=0;
                LDepth1[3]=LDepth1[1] * 32 + LDepth1[2] * 32;
                LDepth1[4]=LDepth1[0] * 32 + LDepth1[2] * 32;
                if(row < (16 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s5];
                    DCTCOUNT_s5++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row < (16 + LDepth1[3]) && col > (15 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=1;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s6];
                    DCTCOUNT_s6++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row > (15 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=1;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s7];
                    DCTCOUNT_s7++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=1;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s8];
                    DCTCOUNT_s8++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
            }
            else if(row > 31 && col < 32)//Depth1
            {
                DCTIN83[DCTCOUNT3] = ptrSrc_5920[col + row * picwid];
                DCTCOUNT3++;
                LDepth1[0]=0;
                LDepth1[1]=1;
                LDepth1[2]=0;
                LDepth1[3]=LDepth1[1] * 32 + LDepth1[2] * 32;
                LDepth1[4]=LDepth1[0] * 32 + LDepth1[2] * 32;
                if(row < (16 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s9];
                    DCTCOUNT_s9++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row < (16 + LDepth1[3]) && col > (15 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=1;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s10];
                    DCTCOUNT_s10++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row > (15 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=1;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s11];
                    DCTCOUNT_s11++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=1;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s12];
                    DCTCOUNT_s12++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
            }
            else//Depth1
            {
                DCTIN84[DCTCOUNT4] = ptrSrc_5920[col + row * picwid];
                DCTCOUNT4++;
                LDepth1[0]=0;
                LDepth1[1]=0;
                LDepth1[2]=1;
                LDepth1[3]=LDepth1[1] * 32 + LDepth1[2] * 32;
                LDepth1[4]=LDepth1[0] * 32 + LDepth1[2] * 32;
                if(row < (16 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s13];
                    DCTCOUNT_s13++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row < (16 + LDepth1[3]) && col > (15 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=1;
                    LDepth2[1]=0;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s14];
                    DCTCOUNT_s14++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else if(row > (15 + LDepth1[3]) && col < (16 + LDepth1[4]))//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=1;
                    LDepth2[2]=0;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s15];
                    DCTCOUNT_s15++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
                else//Depth2
                {
                    LDepth2[0]=0;
                    LDepth2[1]=0;
                    LDepth2[2]=1;
                    DCTIN16x16[LDepth1[0]*4+LDepth1[1]*8+LDepth1[2]*12+LDepth2[0]+LDepth2[1]*2+LDepth2[2]*3][DCTCOUNT_s16];
                    DCTCOUNT_s16++;
                    LDepth2[3]=LDepth2[1] * 16 + LDepth2[2] * 16;
                    LDepth2[4]=LDepth2[0] * 16 + LDepth2[2] * 16;
                    if(row < (8 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row < (8 + LDepth2[3] + LDepth1[3]) && col > (7 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=1;
                        LDepth3[1]=0;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else if(row > (7 + LDepth2[3] + LDepth1[3]) && col < (8 + LDepth2[4] + LDepth1[4]))//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=1;
                        LDepth3[2]=0;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                    else//Depth3
                    {
                        LDepth3[0]=0;
                        LDepth3[1]=0;
                        LDepth3[2]=1;
                        LDepth3[3]=LDepth3[1] * 8 + LDepth3[2] * 8;
                        LDepth3[4]=LDepth3[0] * 8 + LDepth3[2] * 8;
                        INDEX=LDepth1[0] * 64 + LDepth1[1] * 128 + LDepth1[2] * 192 + LDepth2[0] * 16 + LDepth2[1] * 32 + LDepth2[2] * 48 + LDepth3[0] * 4 + LDepth3[1] * 8 + LDepth3[2] * 12;  
                        if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[0+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row < (4 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col > (3 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[1+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else if(row > (3 + LDepth3[3] + LDepth2[3] + LDepth1[3]) && col < (4 + LDepth3[4] + LDepth2[4] + LDepth1[4]))//Depth4
                        {
                            SUMSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[2+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                        else//Depth4
                        {
                            SUMSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid];
                            SUMVSIZE4[3+INDEX] += ptrSrc_5920[col + row * picwid] * ptrSrc_5920[col + row * picwid];
                        }
                    }
                }
            }
        }
    }
	MYxTrMxN(DCTIN81, DCTOUT81, 32, 32);
	MYxTrMxN(DCTIN82, DCTOUT82, 32, 32);
	MYxTrMxN(DCTIN83, DCTOUT83, 32, 32);
	MYxTrMxN(DCTIN84, DCTOUT84, 32, 32);
    for(int step=0;step<16;step++)
    {
	    MYxTrMxN(DCTIN16x16[step], DCTOUT16x16[step], 16, 16);
        DC_Depth2[step]=abs(DCTOUT16x16[step][0]>>17);
        AC_Depth2[step]=0;
        for(int stepi=1;stepi<256;stepi++)
        {
            AC_Depth2[step]=AC_Depth2[step]+abs(DCTOUT16x16[step][stepi]>>17);
        }
    }
	DCTOUT81[0]=DCTOUT81[0]>>17;
	DCTOUT82[0]=DCTOUT82[0]>>17;
	DCTOUT83[0]=DCTOUT83[0]>>17;
	DCTOUT84[0]=DCTOUT84[0]>>17;
	for(int stepi=1;stepi<1024;stepi++)
	{
		DCTOUT81[stepi]=abs(DCTOUT81[stepi]>>17);
		DCTSUM81=DCTSUM81+(double)DCTOUT81[stepi];
		DCTOUT82[stepi]=abs(DCTOUT82[stepi]>>17);
		DCTSUM82=DCTSUM82+(double)DCTOUT82[stepi];
		DCTOUT83[stepi]=abs(DCTOUT83[stepi]>>17);
		DCTSUM83=DCTSUM83+(double)DCTOUT83[stepi];
		DCTOUT84[stepi]=abs(DCTOUT84[stepi]>>17);
		DCTSUM84=DCTSUM84+(double)DCTOUT84[stepi];
	}
	DC_AVG =(double)( DCTOUT81[0] + DCTOUT82[0] + DCTOUT83[0] + DCTOUT84[0] )/ 4.0;
	AC_AVG =(DCTSUM81+DCTSUM82+DCTSUM83+DCTSUM84)/4.0;
    for(int step = 0; step < 4; step++)
        DC_Depth1[step] = DCTOUT81[step];
    AC_Depth1[0] = DCTSUM81;
    AC_Depth1[1] = DCTSUM82;
    AC_Depth1[2] = DCTSUM83;
    AC_Depth1[3] = DCTSUM84;
    for(int step = 0; step < 64; step++)
    {
        AVGofCOL64[step] = AVGofCOL64[step] / 64.0;
        SUMAVGofCOL64 += AVGofCOL64[step];
        SUMVAVGofCOL64 += AVGofCOL64[step] * AVGofCOL64[step];
        AVGofROW64[step] = AVGofROW64[step] / 64.0;
        SUMAVGofROW64 += AVGofROW64[step];
        SUMVAVGofROW64 += AVGofROW64[step] * AVGofROW64[step];
    }
    AVGAVGofCOL64 = SUMAVGofCOL64 / 64.0;
    VARAVGofCOL64 = SUMVAVGofCOL64 / 64.0 - AVGAVGofCOL64 * AVGAVGofCOL64;
    AVGAVGofROW64 = SUMAVGofROW64 / 64.0;
    VARAVGofROW64 = SUMVAVGofROW64 / 64.0 - AVGAVGofROW64 * AVGAVGofROW64;
    for(int d1 = 0; d1 < 4; d1++)//Elevate SUM/SUMV/AVG/VAR64 by adding SUM/SUMV4
    {
        for(int d2 = 0; d2 < 4; d2++)
        {
            for(int d3 = 0; d3 < 4; d3++)
            {
                for(int step = 0; step < 4; step++)
                {
                    SUMSIZE8[d1][d2][d3] += SUMSIZE4[step+d3*4+d2*16+d1*64];
                    SUMVSIZE8[d1][d2][d3] += SUMVSIZE4[step+d3*4+d2*16+d1*64];
                    VARINPUT[0] = SUMSIZE4[step+d3*4+d2*16+d1*64];
                    VARINPUT[1] = SUMVSIZE4[step+d3*4+d2*16+d1*64];
                    Variance(VARINPUT, 16.0, &VARSIZE4[step+d3*4+d2*16+d1*64], &AVGSIZE4[step+d3*4+d2*16+d1*64], VAR_SUM_MODE);
                }

                SUMSIZE16[d1][d2] += SUMSIZE8[d1][d2][d3];
                SUMVSIZE16[d1][d2] += SUMVSIZE8[d1][d2][d3];
                VARINPUT[0] = SUMSIZE8[d1][d2][d3];
                VARINPUT[1] = SUMVSIZE8[d1][d2][d3];
                Variance(VARINPUT, 64.0, &VARSIZE8[d1][d2][d3], &AVGSIZE8[d1][d2][d3], VAR_SUM_MODE);
            }

            SUMSIZE32[d1] += SUMSIZE16[d1][d2];
            SUMVSIZE32[d1] += SUMVSIZE16[d1][d2];
            VARINPUT[0] = SUMSIZE16[d1][d2];
            VARINPUT[1] = SUMVSIZE16[d1][d2];
            Variance(VARINPUT, 256.0, &VARSIZE16[d1][d2], &AVGSIZE16[d1][d2], VAR_SUM_MODE);
        }

        SUMSIZE64 += SUMSIZE32[d1];
        SUMVSIZE64 += SUMVSIZE32[d1];
        VARINPUT[0] = SUMSIZE32[d1];
        VARINPUT[1] = SUMVSIZE32[d1];
        Variance(VARINPUT, 1024.0, &VARSIZE32[d1], &AVGSIZE32[d1], VAR_SUM_MODE);
    }
    VARINPUT[0] = SUMSIZE64;
    VARINPUT[1] = SUMVSIZE64;
    Variance(VARINPUT, 4096.0, &VARSIZE64, &AVGSIZE64, VAR_SUM_MODE);
}

int CUFeature::getINTRADepth0(float* Output,UInt uiAbsPartIdx)
{
    Double VARofVARSIZE32;
    Double AVGofVARSIZE32;
    Double VARofAVGSIZE32;
    Double AVGofAVGSIZE32;
    Variance(VARSIZE32, 4.0, &VARofVARSIZE32, &AVGofVARSIZE32, VAR_NORMAL_MODE);  
    Variance(AVGSIZE32, 4.0, &VARofAVGSIZE32, &AVGofAVGSIZE32, VAR_NORMAL_MODE); 

//T24================================================================================
    Output[0]=VARAVGofCOL64;
    Output[1]=VARAVGofROW64;
    Output[2]=VARofVARSIZE32;
    Output[3]=VARofAVGSIZE32;
    Output[4]=VARSIZE64;
    Output[5]=DC_AVG;
    Output[6]=AC_AVG;
    Output[7]=QP;
    return 8;
//T24================================================================================
//MIXT1================================================================================
    /*Output[0]=QP;
    Output[1]=VARAVGofCOL64;
    Output[2]=VARAVGofROW64;
    Output[3]=VARofVARSIZE32;
    Output[4]=VARofAVGSIZE32;
    Output[5]=VARSIZE64;
    Output[6]=DC_AVG;
    Output[7]=AC_AVG;
    Output[8]=0;
    Output[9]=0;
    Output[10]=0;
    Output[11]=0;
    Output[12]=0;
    Output[13]=0;
    Output[14]=0;
    Output[15]=0;
    Output[16]=64;
    return 17;*/
//MIXT1================================================================================
    /*Output[0]=QP;
    Output[1]=VARSIZE64;
    Output[2]=AVGSIZE64;
    Output[3]=VARofVARSIZE32;
    Output[4]=AVGofVARSIZE32;
    Output[5]=VARofAVGSIZE32;
    Output[6]=AVGofAVGSIZE32;*/
    //Output[8]=VARDiffrenceofPreFrame;
    //Output[9]=AVGDiffrenceofPreFrame;
}

int CUFeature::getINTRADepth1(float* Output,UInt uiAbsPartIdx)
{
    Double VARofVARSIZE16;
    Double AVGofVARSIZE16;
    Double VARofAVGSIZE16;
    Double AVGofAVGSIZE16;
    int INDEX;
    INDEX=uiAbsPartIdx/64;
    Variance(VARSIZE16[INDEX], 4.0, &VARofVARSIZE16, &AVGofVARSIZE16, VAR_NORMAL_MODE);  
    Variance(AVGSIZE16[INDEX], 4.0, &VARofAVGSIZE16, &AVGofAVGSIZE16, VAR_NORMAL_MODE);  
//T26================================================================================
    Output[0]=VARofVARSIZE16;
    Output[1]=VARSIZE32[INDEX];
    Output[2]=QP;
    return 3;
//T26================================================================================
//MIXT1================================================================================
    /*Output[0]=QP;
    Output[1]=0;
    Output[2]=0;
    Output[3]=0;
    Output[4]=0;
    Output[5]=0;
    Output[6]=0;
    Output[7]=0;
    Output[8]=VARofVARSIZE16;
    Output[9]=VARSIZE32[INDEX];
    Output[10]=0;
    Output[11]=0;
    Output[12]=0;
    Output[13]=0;
    Output[14]=0;
    Output[15]=0;
    Output[16]=32;
    return 17;*/
//MIXT1================================================================================
    /*Output[0]=QP;
    Output[1]=VARSIZE32[INDEX];
    Output[2]=AVGSIZE32[INDEX];
    Output[3]=VARofVARSIZE16;
    Output[4]=AVGofVARSIZE16;
    Output[5]=VARofAVGSIZE16;
    Output[6]=AVGofAVGSIZE16;*/
    //Output[3]=VARDiffrenceofPreFrame;
    //Output[4]=AVGDiffrenceofPreFrame;
}

int CUFeature::getINTRADepth2(float* Output,UInt uiAbsPartIdx)
{
    Double VARofVARSIZE8;
    Double AVGofVARSIZE8;
    Double VARofAVGSIZE8;
    Double AVGofAVGSIZE8;
    int INDEX,INDEXNEXT;
    INDEX=uiAbsPartIdx/64;
    INDEXNEXT=(uiAbsPartIdx%64)/16;
    Variance(VARSIZE8[INDEX][INDEXNEXT], 4.0, &VARofVARSIZE8, &AVGofVARSIZE8, VAR_NORMAL_MODE);  
    Variance(AVGSIZE8[INDEX][INDEXNEXT], 4.0, &VARofAVGSIZE8, &AVGofAVGSIZE8, VAR_NORMAL_MODE);  
//MIXT1================================================================================
    /*Output[0]=QP;
    Output[1]=0;
    Output[2]=0;
    Output[3]=0;
    Output[4]=0;
    Output[5]=0;
    Output[6]=0;
    Output[7]=0;
    Output[8]=0;
    Output[9]=0;
    Output[10]=VARSIZE16[INDEX][INDEXNEXT];
    Output[11]=AVGSIZE16[INDEX][INDEXNEXT];
    Output[12]=VARofVARSIZE8;
    Output[13]=AVGofVARSIZE8;
    Output[14]=VARofAVGSIZE8;
    Output[15]=AVGofAVGSIZE8;
    Output[16]=16;
    return 17;*/
//MIXT1================================================================================
    Output[0]=QP;
    Output[1]=VARSIZE16[INDEX][INDEXNEXT];
    Output[2]=AVGSIZE16[INDEX][INDEXNEXT];
    Output[3]=VARofVARSIZE8;
    Output[4]=AVGofVARSIZE8;
    Output[5]=VARofAVGSIZE8;
    Output[6]=AVGofAVGSIZE8;
    //Output[7]=VARDiffrenceofPreFrame;
    //Output[8]=AVGDiffrenceofPreFrame;
    return 7;
}
int CUFeature::getINTERDepth0(float* Output,UInt uiAbsPartIdx)
{
    Double VARofVARSIZE32;
    Double AVGofVARSIZE32;
    Double VARofAVGSIZE32;
    Double AVGofAVGSIZE32;
    Variance(VARSIZE32, 4.0, &VARofVARSIZE32, &AVGofVARSIZE32, VAR_NORMAL_MODE);  
    Variance(AVGSIZE32, 4.0, &VARofAVGSIZE32, &AVGofAVGSIZE32, VAR_NORMAL_MODE); 
//INTERT09================================================================================
    Output[0]=VARAVGofCOL64;
    Output[1]=VARAVGofROW64;
    Output[2]=VARofVARSIZE32;
    Output[3]=VARofAVGSIZE32;
    Output[4]=VARSIZE64;
    Output[5]=DC_AVG;
    Output[6]=AC_AVG;
    Output[7]=QP;
    Output[8]=DIFFPREVARSIZE64;
    Output[9]=DIFFPREAVGSIZE64;
    return 10;
//INTERT09================================================================================
}

int CUFeature::getINTERDepth1(float* Output,UInt uiAbsPartIdx)
{
    Double VARofVARSIZE16;
    Double AVGofVARSIZE16;
    Double VARofAVGSIZE16;
    Double AVGofAVGSIZE16;
    int INDEX;
    INDEX=uiAbsPartIdx/64;
    Variance(VARSIZE16[INDEX], 4.0, &VARofVARSIZE16, &AVGofVARSIZE16, VAR_NORMAL_MODE);  
    Variance(AVGSIZE16[INDEX], 4.0, &VARofAVGSIZE16, &AVGofAVGSIZE16, VAR_NORMAL_MODE);  
//INTERT09================================================================================
    Output[0]=VARofVARSIZE16;
    Output[1]=VARSIZE32[INDEX];
    Output[2]=QP;
    Output[3]=DIFFPREVARSIZE64;
    Output[4]=DIFFPREAVGSIZE64;
    return 5;
//INTERT09================================================================================
}

int CUFeature::getINTERDepth2(float* Output,UInt uiAbsPartIdx)
{
    Double VARofVARSIZE8;
    Double AVGofVARSIZE8;
    Double VARofAVGSIZE8;
    Double AVGofAVGSIZE8;
    int INDEX,INDEXNEXT;
    INDEX=uiAbsPartIdx/64;
    INDEXNEXT=(uiAbsPartIdx%64)/16;
    Variance(VARSIZE8[INDEX][INDEXNEXT], 4.0, &VARofVARSIZE8, &AVGofVARSIZE8, VAR_NORMAL_MODE);  
    Variance(AVGSIZE8[INDEX][INDEXNEXT], 4.0, &VARofAVGSIZE8, &AVGofAVGSIZE8, VAR_NORMAL_MODE);  
//INTERT09================================================================================
    Output[0]=QP;
    Output[1]=VARSIZE16[INDEX][INDEXNEXT];
    Output[2]=AVGSIZE16[INDEX][INDEXNEXT];
    Output[3]=VARofVARSIZE8;
    Output[4]=AVGofVARSIZE8;
    Output[5]=VARofAVGSIZE8;
    Output[6]=AVGofAVGSIZE8;
    Output[7]=DIFFPREVARSIZE64;
    Output[8]=DIFFPREAVGSIZE64;
    return 9;
//INTERT09================================================================================
}
int CUFeature::getPMDepth0(float* Output,UInt uiAbsPartIdx)
{
    Double VARofVARSIZE32;
    Double AVGofVARSIZE32;
    Double VARofAVGSIZE32;
    Double AVGofAVGSIZE32;
    Variance(VARSIZE32, 4.0, &VARofVARSIZE32, &AVGofVARSIZE32, VAR_NORMAL_MODE);  
    Variance(AVGSIZE32, 4.0, &VARofAVGSIZE32, &AVGofAVGSIZE32, VAR_NORMAL_MODE); 
//INTERT09================================================================================
    Output[0]=VARAVGofCOL64;
    Output[1]=VARAVGofROW64;
    Output[2]=VARofVARSIZE32;
    Output[3]=VARofAVGSIZE32;
    Output[4]=VARSIZE64;
    Output[5]=DC_AVG;
    Output[6]=AC_AVG;
    Output[7]=QP;
    Output[8]=DIFFPREVARSIZE64;
    Output[9]=DIFFPREAVGSIZE64;
    return 10;
//INTERT09================================================================================
}

int CUFeature::getPMDepth1(float* Output,UInt uiAbsPartIdx)
{
    Double VARofVARSIZE16;
    Double AVGofVARSIZE16;
    Double VARofAVGSIZE16;
    Double AVGofAVGSIZE16;
    int INDEX;
    INDEX=uiAbsPartIdx/64;
    Variance(VARSIZE16[INDEX], 4.0, &VARofVARSIZE16, &AVGofVARSIZE16, VAR_NORMAL_MODE);  
    Variance(AVGSIZE16[INDEX], 4.0, &VARofAVGSIZE16, &AVGofAVGSIZE16, VAR_NORMAL_MODE);  
//INTERT09================================================================================
    Output[0]=VARofVARSIZE16;
    Output[1]=VARSIZE32[INDEX];
    Output[2]=QP;
    Output[3]=DIFFPREVARSIZE32[INDEX];
    Output[4]=DIFFPREAVGSIZE32[INDEX];
    Output[5]=VARofAVGSIZE16;
    Output[6]=DC_Depth1[INDEX];
    Output[7]=AC_Depth1[INDEX];
    return 8;
//INTERT09================================================================================
}

int CUFeature::getPMDepth2(float* Output,UInt uiAbsPartIdx)
{
    Double VARofVARSIZE8;
    Double AVGofVARSIZE8;
    Double VARofAVGSIZE8;
    Double AVGofAVGSIZE8;
    int INDEX,INDEXNEXT;
    INDEX=uiAbsPartIdx/64;
    INDEXNEXT=(uiAbsPartIdx%64)/16;
    Variance(VARSIZE8[INDEX][INDEXNEXT], 4.0, &VARofVARSIZE8, &AVGofVARSIZE8, VAR_NORMAL_MODE);  
    Variance(AVGSIZE8[INDEX][INDEXNEXT], 4.0, &VARofAVGSIZE8, &AVGofAVGSIZE8, VAR_NORMAL_MODE);  
//INTERT09================================================================================
    Output[0]=QP;
    Output[1]=VARSIZE16[INDEX][INDEXNEXT];
    Output[2]=AVGSIZE16[INDEX][INDEXNEXT];
    Output[3]=VARofVARSIZE8;
    Output[4]=AVGofVARSIZE8;
    Output[5]=VARofAVGSIZE8;
    Output[6]=AVGofAVGSIZE8;
    Output[7]=DIFFPREVARSIZE16[INDEX][INDEXNEXT];
    Output[8]=DIFFPREAVGSIZE16[INDEX][INDEXNEXT];
    Output[9]=DC_Depth2[INDEX*4+INDEXNEXT];
    Output[10]=AC_Depth2[INDEX*4+INDEXNEXT];
    return 11;
//INTERT09================================================================================
}
int CUFeature::getFeature(float* Output,int Depth,UInt uiAbsPartIdx,int Mode)
{
    int NUM;
    if(Mode==FEATURE_INTRAMODE)
    {
        switch(Depth)
        {
            case 0:
                NUM=getINTRADepth0(Output,uiAbsPartIdx);
                break;
            case 1:
                NUM=getINTRADepth1(Output,uiAbsPartIdx);
                break;
            case 2:
                NUM=getINTRADepth2(Output,uiAbsPartIdx);
                break;
        }
    }
    if(Mode==FEATURE_INTERMODE)
    {
        switch(Depth)
        {
            case 0:
                NUM=getINTERDepth0(Output,uiAbsPartIdx);
                break;
            case 1:
                NUM=getINTERDepth1(Output,uiAbsPartIdx);
                break;
            case 2:
                NUM=getINTERDepth2(Output,uiAbsPartIdx);
                break;
        }
    }
    if(Mode==FEATURE_PREDICTMODE)
    {
        switch(Depth)
        {
            case 0:
                NUM=getPMDepth0(Output,uiAbsPartIdx);
                break;
            case 1:
                NUM=getPMDepth1(Output,uiAbsPartIdx);
                break;
            case 2:
                NUM=getPMDepth2(Output,uiAbsPartIdx);
                break;
        }
    }
    return NUM;
}

void CUFeature::ElevateINTERDiff(TComDataCU* pcCU)
{
    if(INTERPRE_READY)
    {
        DIFFPREVARSIZE64=abs(PREVARSIZE64-VARSIZE64);
        DIFFPREAVGSIZE64=abs(PREAVGSIZE64-AVGSIZE64);

        for(int INDEX=0;INDEX<4;INDEX++)
        {
            DIFFPREAVGSIZE32[INDEX]=abs(PREAVGSIZE32[INDEX]-AVGSIZE32[INDEX]);
            DIFFPREVARSIZE32[INDEX]=abs(PREVARSIZE32[INDEX]-VARSIZE32[INDEX]);
        }
        for(int INDEX=0;INDEX<4;INDEX++)
        {
            for(int INDEX2=0;INDEX2<4;INDEX2++)
            {
                DIFFPREAVGSIZE16[INDEX][INDEX2]=abs(PREAVGSIZE16[INDEX][INDEX2]-AVGSIZE16[INDEX][INDEX2]);
                DIFFPREVARSIZE16[INDEX][INDEX2]=abs(PREVARSIZE16[INDEX][INDEX2]-VARSIZE16[INDEX][INDEX2]);
            }
        }
        //VARDiffrenceofPreFrame=PREVARSIZE64;
        //AVGDiffrenceofPreFrame=PREAVGSIZE64;
    }
    //else
        //QP = pcCU->getQP(0);
    PREAVGSIZE64=AVGSIZE64;
    PREVARSIZE64=VARSIZE64;
    for(int INDEX=0;INDEX<4;INDEX++)
    {
        PREAVGSIZE32[INDEX]=AVGSIZE32[INDEX];
        PREVARSIZE32[INDEX]=VARSIZE32[INDEX];
    }
    for(int INDEX=0;INDEX<4;INDEX++)
    {
        for(int INDEX2=0;INDEX2<4;INDEX2++)
        {
            PREAVGSIZE16[INDEX][INDEX2]=AVGSIZE16[INDEX][INDEX2];
            PREVARSIZE16[INDEX][INDEX2]=VARSIZE16[INDEX][INDEX2];
        }
    }
    INTERPRE_READY=true;
}
void CUFeature::SYSTEM_INIT(int iQP)
{
    QP = iQP;
    INTERPRE_READY=false;
}