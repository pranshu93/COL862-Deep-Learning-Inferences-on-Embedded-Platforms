#include <Bonsai_class_fp.h>

bool FLAG = true;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
     while(FLAG)
    {      
      float ***W_fp, ***V_fp, **Z_fp, **Theta_fp, *Score_fp, *Original_Data_Point_fp, *Projected_Data_Point_fp, *mean, *stdd;
      Z_fp = (float**)malloc(Projected_Dimension*sizeof(float*));
      for(i=0;i<Projected_Dimension;i++)
        Z_fp[i] = (float*)malloc(Original_Dimension*sizeof(float));
      
      W_fp = (float***)malloc(Total_Nodes*sizeof(float**));
      for(i=0;i<Total_Nodes;i++)
      {
        W_fp[i] = (float**)malloc(Projected_Dimension*sizeof(float*));
        for(j=0;j<Projected_Dimension;j++)
            W_fp[i][j] = (float*)malloc(Number_Of_Labels*sizeof(float));
      }
      
      V_fp = (float***)malloc(Total_Nodes*sizeof(float**));
      for(i=0;i<Total_Nodes;i++)
      {
        V_fp[i] = (float**)malloc(Projected_Dimension*sizeof(float*));
        for(j=0;j<Projected_Dimension;j++)
            V_fp[i][j] = (float*)malloc(Number_Of_Labels*sizeof(float));
      }
      
      Theta_fp = (float**)malloc(Internal_Nodes*sizeof(float*));
      for(i=0;i<Internal_Nodes;i++)
        Theta_fp[i] = (float*)malloc(Projected_Dimension*sizeof(float));
      
      Score_fp = (float*)malloc(Number_Of_Labels*sizeof(float));
      memset(Score_fp,0,Number_Of_Labels*sizeof(float));
      mean = (float*)malloc(Original_Dimension*sizeof(float*));
      stdd = (float*)malloc(Original_Dimension*sizeof(float*));
      Original_Data_Point_fp = (float*)malloc(Original_Dimension*sizeof(float*));
          
      for(i=0;i<Internal_Nodes;i++)
      {
        for(j=0;j<Projected_Dimension;j++)
            Theta_fp[i][j] = buffer[ctr++];              
      }
      
      for(i=0;i<Total_Nodes;i++)
      {
        for(j=0;j<Projected_Dimension;j++)
        {
            for(k=0;k<Number_Of_Labels;k++)
                W_fp[i][j][k] = buffer[ctr++];                  
        }
      }
      
      for(i=0;i<Total_Nodes;i++)
      {
        for(j=0;j<Projected_Dimension;j++)
        {
            for(k=0;k<Number_Of_Labels;k++)
                V_fp[i][j][k] = buffer[ctr++];                  
        }
      }
      
      for(i=0;i<Projected_Dimension;i++)
      {
        for(j=0;j<Original_Dimension;j++)
            Z_fp[i][j] = buffer[ctr++];              
      }
      
      for(i=0;i<Original_Dimension;i++)
        mean[i] = buffer[ctr++];
      
      for(i=0;i<Original_Dimension;i++)
        stdd[i] = buffer[ctr++];
       
      for(i=0;i<Original_Dimension;i++)
        Original_Data_Point_fp[i] = buffer[ctr++];
      
      for(i=0;i<Original_Dimension;i++)
        Original_Data_Point_fp[i] = (Original_Data_Point_fp[i] - mean[i])/(stdd[i]);      
      
      Bonsai_fp Sample(Projected_Dimension,Original_Dimension,Tree_Height,Number_Of_Labels, sigma, W_fp,V_fp,Z_fp,Theta_fp);
      Sample.Get_Score(Original_Data_Point_fp,true);
      
      FLAG = false;
      
    }
  
}


