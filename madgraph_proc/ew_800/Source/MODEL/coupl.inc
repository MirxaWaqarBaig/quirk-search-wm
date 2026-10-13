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
      PARAMETER(NF=5D0)
      DOUBLE PRECISION NL
      PARAMETER(NL=3D0)

      DOUBLE PRECISION MDL_MH,MDL_MT,MDL_MTAUP,MDL_MW,MDL_MZ

      COMMON/MASSES/ MDL_MH,MDL_MT,MDL_MTAUP,MDL_MW,MDL_MZ


      DOUBLE PRECISION MDL_WH,MDL_WT,MDL_WTAUP,MDL_WW,MDL_WZ

      COMMON/WIDTHS/ MDL_WH,MDL_WT,MDL_WTAUP,MDL_WW,MDL_WZ


      DOUBLE COMPLEX GC_1, GC_19, GC_2, GC_48, GC_49, GC_51, GC_53,
     $  GC_54

      DOUBLE COMPLEX GC_15(VECSIZE_MEMMAX)

      COMMON/COUPLINGS/ GC_15, GC_1, GC_19, GC_2, GC_48, GC_49, GC_51,
     $  GC_53, GC_54

