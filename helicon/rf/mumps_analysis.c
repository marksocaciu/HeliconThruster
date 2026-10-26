/* Small public-PETSc-API bridge: petsc4py symbolic LU is not implemented.
   Compiled against the selected complex PETSc, never a second solver runtime. */
#include <petscmat.h>
int helicon_petsc_abi(int *values) {
  values[0]=PETSC_VERSION_MAJOR; values[1]=PETSC_VERSION_MINOR;
  values[2]=PETSC_VERSION_SUBMINOR; values[3]=sizeof(PetscInt);
  values[4]=sizeof(PetscScalar); return 0;
}
PetscErrorCode helicon_mumps_analysis(Mat A, const char *prefix, long long *out) {
  Mat F=NULL;
  MatFactorInfo info;
  PetscErrorCode err=0, cleanup;
  PetscInt value;
  int keys[]={1,2,7,16,17,26,27,38,39};
#define TRY(call) do {err=(call); if(err) goto done;} while(0)
  TRY(MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_LU,&F));
  TRY(MatSetOptionsPrefix(F,prefix));
  TRY(MatSetErrorIfFailure(F,PETSC_TRUE));
  TRY(MatFactorInfoInitialize(&info));
  TRY(MatLUFactorSymbolic(F,A,NULL,NULL,&info));
  for(int i=0;i<9;i++) { TRY(MatMumpsGetInfog(F,keys[i],&value)); out[i]=(long long)value; }
  TRY(MatMumpsGetInfo(F,17,&value)); out[9]=(long long)value;
  TRY(MatMumpsGetInfo(F,31,&value)); out[10]=(long long)value;
  TRY(MatMumpsGetInfo(F,15,&value)); out[11]=(long long)value;
  TRY(MatMumpsGetInfo(F,30,&value)); out[12]=(long long)value;
done:
  cleanup=MatDestroy(&F);
  return err ? err : cleanup;
}
