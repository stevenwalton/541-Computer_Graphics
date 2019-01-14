#include <iostream>
#include <vtkDataSet.h>
#include <vtkImageData.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>

#include <algorithm>

using std::cerr;
using std::endl;

double ceil_441(double f)
{
    return ceil(f-0.00001);
}

double floor_441(double f)
{
    return floor(f+0.00001);
}


vtkImageData *
NewImage(int width, int height)
{
    vtkImageData *img = vtkImageData::New();
    img->SetDimensions(width, height, 1);
    img->AllocateScalars(VTK_UNSIGNED_CHAR, 3);

    return img;
}

void
WriteImage(vtkImageData *img, const char *filename)
{
   std::string full_filename = filename;
   full_filename += ".png";
   vtkPNGWriter *writer = vtkPNGWriter::New();
   writer->SetInputData(img);
   writer->SetFileName(full_filename.c_str());
   writer->Write();
   writer->Delete();
}

class Triangle
{
  public:
      double         X[3];
      double         Y[3];
      unsigned char  color[3];
      // bounding box
      int            index_max;
      int            index_min;
      double         minX;
      double         maxX;
      double         minY;
      double         maxY;
      void           getMinMax();

      void           adjustTriangle();
      bool           isRight(); 
      bool           checkVert() {return (X[0] == X[1] || X[0] == X[2] || X[1] == X[2]);}
      //// For bounding box

      void           writePixel(unsigned char *buffer, int index);
      void           drawTriangle(unsigned char *buffer, int width, int height);

  // would some methods for transforming the triangle in place be helpful?
};

void
Triangle::getMinMax()
{
    minX = *std::min_element(X,X+3);
    maxX = *std::max_element(X,X+3);
    minY = *std::min_element(Y,Y+3);
    maxY = *std::max_element(Y,Y+3);
    index_max = 0;
    index_min = 0;
    for (int i = 0; i < 3; ++i)
    {
        if(X[i] == minX && Y[i] == minY) 
            index_min = i;
        else if(X[i] == maxX && Y[i] == maxY)
            index_max = i;
    }
}

void
Triangle::adjustTriangle()
{
    for(int i = 0; i < 3; ++i)
    {
        if( i == index_min)
        {
            X[i] = ceil_441(X[i]);
            Y[i] = ceil_441(Y[i]);
        }
        else if (i == index_max)
        {
            X[i] = floor_441(X[i]);
            Y[i] = floor_441(Y[i]);
        }
    }
}

bool
Triangle::isRight()
{
    if ((X[0] == X[1] || X[0] == X[2] || X[1] == X[2]) &&
        (Y[0] == Y[1] || Y[0] == Y[2] || Y[1] == Y[2]))
            return true;
    return false;
}

void
Triangle::writePixel(unsigned char *buffer, int index)
{
    buffer[index+0] = color[0];
    buffer[index+1] = color[1];
    buffer[index+2] = color[2];
}

void
Triangle::drawTriangle(unsigned char *buffer, int width, int height)
{
    double m0, m1, y0, y1, b0, b1;
    // Our points
    // The triangles are just rotations of one another, so we can just rotate
    // our points to draw the triangles with the same method
    int i = 0;
    int j = 1;
    int k = 2;
    // Do our rotations
    if(Y[0] == Y[1]) // Greens and oranges
    {
        i = 0;
        j = 1;
        k = 2;
    }
    else if(Y[1] == Y[2]) // Pinks and grays
    {
        i = 1;
        j = 2;
        k = 0;
    }
    
    else //if(Y[0] == Y[2]) // Yellow and Cyan
    {
        i = 2;
        j = 0;
        k = 1;
    }
    m0 = (Y[j]-Y[k])/(X[j]-X[k]);
    m1 = (Y[k]-Y[i])/(X[k]-X[i]);
    b0 = Y[k] - (m0*X[k]);
    b1 = Y[i] - (m1*X[i]);
    // Adjust our bounds for screen size
    if(minX < 0) minX = 0;
    if(maxX >= width) maxX = width-1;
    if(minY < 0) minY = 0;
    if(maxY >= height) maxY = height-1;
    // Loop over bounding box
    for(int col = minX; col <= maxX; ++col)
        for(int row = minY; row <= maxY; ++row)
        {
            int index = ((row*width) + col)*3;
            if (isRight())
            {
                if (X[1] == X[2])
                {
                    // Top right
                    y0 = Y[k];
                    y1 = m1*col + b1;
                }
                else if (X[0] == X[2])
                {
                    // Bottom left
                    y0 = m0*col + b0;
                    y1 = Y[k];
                }
            }
            else
            {
                y0 = m0*col + b0;
                y1 = m1*col + b1;
            }

            if((col >= (X[i]) && row >= ceil_441(y0)) && 
               (col <= (X[j]) && row >= ceil_441(y1)))
            {
                writePixel(buffer,index);
            }
        }

}


class Screen
{
  public:
      unsigned char   *buffer;
      int width, height;

  // would some methods for accessing and setting pixels be helpful?
};

std::vector<Triangle>
GetTriangles(void)
{
   std::vector<Triangle> rv(100);

   unsigned char colors[6][3] = { {255,128,0}, {255, 0, 127}, {0,204,204}, 
                                  {76,153,0}, {255, 204, 204}, {204, 204, 0}};
   for (int i = 0 ; i < 100 ; i++)
   {
       int idxI = i%10;
       int posI = idxI*100;
       int idxJ = i/10;
       int posJ = idxJ*100;
       int firstPt = (i%3);
       rv[i].X[firstPt] = posI;
       if (i == 50)
           rv[i].X[firstPt] = -10;
       rv[i].Y[firstPt] = posJ+10*(idxJ+1);
       rv[i].X[(firstPt+1)%3] = posI+99;
       rv[i].Y[(firstPt+1)%3] = posJ+10*(idxJ+1);
       rv[i].X[(firstPt+2)%3] = posI+i;
       rv[i].Y[(firstPt+2)%3] = posJ;
       if (i == 5)
          rv[i].Y[(firstPt+2)%3] = -50;
       rv[i].color[0] = colors[i%6][0];
       rv[i].color[1] = colors[i%6][1];
       rv[i].color[2] = colors[i%6][2];

   }

   return rv;
}

int main()
{
   vtkImageData *image = NewImage(1000, 1000);
   unsigned char *buffer = 
     (unsigned char *) image->GetScalarPointer(0,0,0);
   int npixels = 1000*1000;
   // Initialize everything as black
   for (int i = 0 ; i < npixels*3 ; i++)
       buffer[i] = 0;

   std::vector<Triangle> triangles = GetTriangles();
   
   Screen screen;
   screen.buffer = buffer;
   screen.width = 1000;
   screen.height = 1000;

   // YOUR CODE GOES HERE TO DEPOSIT THE COLORS FROM TRIANGLES 
   // INTO PIXELS USING THE SCANLINE ALGORITHM
   for (int i = 0; i < 100; ++i)
   {
       triangles[i].getMinMax();
       triangles[i].adjustTriangle();
       triangles[i].drawTriangle(buffer, screen.width, screen.height);
   }
   WriteImage(image, "allTriangles");
}
