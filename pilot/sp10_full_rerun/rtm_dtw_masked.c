#include <math.h>
#include <stdlib.h>
/* masked DTW, same recurrence/normalisation as pilot_pipeline.dtw_distance: frame cost = sqrt(sum over jointly observed points of squared dist * J/nobs), 0 if nobs==0; result / (n+m). a: (n,J,2) float32, ma: (n,J) uint8 */
double dtw_masked(const float*a,const unsigned char*ma,const float*b,const unsigned char*mb,int n,int m,int J){
  double*C=(double*)malloc(sizeof(double)*(n+1)*(m+1)); int W=m+1;
  for(int i=0;i<=n;i++)for(int j=0;j<=m;j++)C[i*W+j]=1e300; C[0]=0;
  for(int i=1;i<=n;i++)for(int j=1;j<=m;j++){
    double s=0;int no=0;
    for(int p=0;p<J;p++){ if(ma[(i-1)*J+p]&&mb[(j-1)*J+p]){ double dx=a[((i-1)*J+p)*2]-b[((j-1)*J+p)*2], dy=a[((i-1)*J+p)*2+1]-b[((j-1)*J+p)*2+1]; s+=dx*dx+dy*dy; no++; } }
    double c = no? sqrt(s*((double)J/no)) : 0.0;
    double x=C[(i-1)*W+j],y=C[i*W+j-1],z=C[(i-1)*W+j-1];double mn=x<y?x:y;mn=mn<z?mn:z;C[i*W+j]=c+mn;}
  double r=C[n*W+m];free(C);return r/(n+m);}
