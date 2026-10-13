ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

C     
C     NB: VECSIZE_MEMMAX is defined in vector.inc
C     NB: vector.inc must be included before coupl.inc
C     

      DOUBLE PRECISION G, ALL_G(1)
      COMMON/STRONG/ G, ALL_G

      DOUBLE COMPLEX GAL(2)
      COMMON/WEAK/ GAL

      DOUBLE PRECISION MU_R, ALL_MU_R(1)
      COMMON/RSCALE/ MU_R, ALL_MU_R

      DOUBLE PRECISION NF
      PARAMETER(NF=4D0)
      DOUBLE PRECISION NL
      PARAMETER(NL=2D0)

      DOUBLE PRECISION MDL_MB,MDL_MBP,MDL_MH,MDL_MT,MDL_MTA,MDL_MTP
     $ ,MDL_MW,MDL_MX,MDL_MY,MDL_MZ

      COMMON/MASSES/ MDL_MB,MDL_MBP,MDL_MH,MDL_MT,MDL_MTA,MDL_MTP
     $ ,MDL_MW,MDL_MX,MDL_MY,MDL_MZ


      DOUBLE PRECISION MDL_WBP,MDL_WH,MDL_WT,MDL_WTP,MDL_WW,MDL_WX
     $ ,MDL_WY,MDL_WZ

      COMMON/WIDTHS/ MDL_WBP,MDL_WH,MDL_WT,MDL_WTP,MDL_WW,MDL_WX
     $ ,MDL_WY,MDL_WZ


      DOUBLE COMPLEX GC_6(VECSIZE_MEMMAX), GC_7(VECSIZE_MEMMAX),
     $  GC_8(VECSIZE_MEMMAX)

      COMMON/COUPLINGS/ GC_6, GC_7, GC_8

