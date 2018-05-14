#include <EigenAVR.h>
#include <Eigen30.h>

#include <Bonsai_class_Project.h>

bool FLAG = true;
bool MODE = false;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
     while(FLAG)
    { 
      if(MODE)
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
      }
      else
      {
        ArrayXXf *W,*V,Z;

        Z = ArrayXXf(Projected_Dimension, Original_Dimension);
  
        ArrayXf *Theta,Score,Original_Data_Point,Projected_Data_Point,mean,stdd;
        Score = ArrayXf(Number_Of_Labels);
        Original_Data_Point = ArrayXf(Original_Dimension);
        mean = ArrayXf(Original_Dimension);
        stdd = ArrayXf(Original_Dimension);
        
        W = new ArrayXXf[Total_Nodes];
        V = new ArrayXXf[Total_Nodes];
        Theta = new ArrayXf[Internal_Nodes];  
    
        for(i=0;i<Total_Nodes;i++)
        {
            W[i] = ArrayXXf(Projected_Dimension,Number_Of_Labels);
            V[i] = ArrayXXf(Projected_Dimension,Number_Of_Labels);
        }
        for(i=0;i<Internal_Nodes;i++)
            Theta[i] = ArrayXf(Projected_Dimension);
        
        
        for(i=0;i<Internal_Nodes;i++)
        {
            for(j=0;j<Projected_Dimension;j++)
                Theta[i](j) = buffer[ctr++];              
        }
        
        for(i=0;i<Total_Nodes;i++)
        {
            for(j=0;j<Projected_Dimension;j++)
            {
                for(k=0;k<Number_Of_Labels;k++)
                    W[i](j,k) = buffer[ctr++];                  
            }
        }
    
        for(i=0;i<Total_Nodes;i++)
        {
            for(j=0;j<Projected_Dimension;j++)
            {
                for(k=0;k<Number_Of_Labels;k++)
                    V[i](j,k) = buffer[ctr++];                  
            }
        }
    
        for(i=0;i<Projected_Dimension;i++)
        {
            for(j=0;j<Original_Dimension;j++)
                Z(i,j) = buffer[ctr++];              
        }        
    
        for(i=0;i<Original_Dimension;i++)
            mean(i) = buffer[ctr++];

        for(i=0;i<Original_Dimension;i++)
            stdd(i) = buffer[ctr++];

        for(i=0;i<Original_Dimension;i++)
            Original_Data_Point(i) = buffer[ctr++];

        Original_Data_Point = Original_Data_Point - mean;
            
        Serial.println();
        Bonsai_Eigen tr(Projected_Dimension,Original_Dimension,Tree_Height,Number_Of_Labels, sigma,W,V,Z,Theta);
        tr.Get_Score(Original_Data_Point,true);
 
      }
      FLAG = false;      
    }  
}


