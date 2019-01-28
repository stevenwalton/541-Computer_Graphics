#include <iostream>
#include <vtkDataSet.h>
#include <vtkImageData.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkFloatArray.h>
#include <vtkCellArray.h>

#include <algorithm>
#include <cmath>

using std::cerr;
using std::endl;
// Screen size globals
static int Nx = 1000;
static int Ny = 1000;

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

class Screen 
{
  public:
      // Constructor
      Screen(int w, int h) {width = w; height = h;}
      // Values
      unsigned char   *buffer;
      int             width, height;
      // Functions
      void SetPixel(int c, int r, unsigned char *color);
};

void
Screen::SetPixel(int c, int r, unsigned char *color)
{
    if (r < 0 || r >= height || c < 0 || c>= width)
        return;
    int index = ((r*width) + c)*3;
    buffer[index+0] = color[0];
    buffer[index+1] = color[1];
    buffer[index+2] = color[2];
}

// Make a global object
Screen screen(Nx, Ny);

class Triangle
{
    public:
        // Values
        double          X[3];
        double          Y[3];
        double          Z[3];
        unsigned char   colors[3][3];
        double          minX, maxX, minY, maxY;
        int             minY_index, maxY_index, middleY_index;
        // Functions
        void            raster();
        void            getMinMax();
        double          getXIntercept();
        void            splitTriangles(); 
        void            drawTriangle(double lX[3], double lY[3]); 
};

void
Triangle::raster()
{
    getMinMax();
    splitTriangles();
}

void
Triangle::getMinMax()
{
    minX = *std::min_element(X,X+3);
    maxX = *std::max_element(X,X+3);
    
    minY = *std::min_element(Y,Y+3);
    maxY = *std::max_element(Y,Y+3);
    
    for(int i = 0; i < 3; ++i)
    {
        if(Y[i] == minY) minY_index = i;
        else if(Y[i] == maxY) maxY_index = i;
        else middleY_index = i;
    }
}

double
Triangle::getXIntercept()
{
    double m = (Y[maxY_index] - Y[minY_index]) / (X[maxY_index] - X[minY_index]);
    double b = Y[maxY_index] - (m * X[maxY_index]);
    return ( (Y[middleY_index] - b) / m);
}

void
Triangle::splitTriangles()
{
    // Set X in correct order for easier math
    double xIntercept = getXIntercept();
    // Lower Triangle
    double lX[3] = {X[middleY_index], xIntercept, X[minY_index]};
    double lY[3] = {Y[middleY_index], Y[middleY_index], minY};
    //drawTriangle(lX, lY);
    // Upper triangle
    double uX[3] = {X[middleY_index], xIntercept, X[maxY_index]};
    double uY[3] = {Y[middleY_index], Y[middleY_index], maxY};
    //drawTriangle(uX, uY);
}

void
Triangle::drawTriangle(double lX[3], double lY[3])
{
    double lminX = *std::min_element(lX, lX+3);
    double lmaxX = *std::max_element(lX, lX+3);

    double lminY = *std::min_element(lY, lY+3);
    double lmaxY = *std::max_element(lY, lY+3);
    // Adjust triangle
    if(lX[0] > lX[1])
    {
        double x0,x1;
        x0 = lX[1];
        x1 = lX[0];
        lX[0] = x0;
        lX[1] = x1;
    }
    double m0,m1,b0,b1,y0,y1;
    if(lX[0] != lX[2]) // right triangle
    {
        m0 = (lY[2] - lY[0]) / (lX[2] - lX[0]);
        b0 = lY[0] - (m0*lX[0]);
    }
    if(lX[1] != lX[2]) // right triangle
    {
        m1 = (lY[2] - lY[1]) / (lX[2] - lX[1]);
        b1 = lY[1] - (m1*lX[1]);
    }
    for(int row = ceil_441(lminY); row <= floor_441(lmaxY); ++row)
    {
        double leftX, rightX;
        leftX = (row - b0)/m0;
        rightX = (row - b1)/m1;
        // Only loop over x pixels within triangle
        //for(int col = ceil_441(leftX); col <= floor_441(rightX); ++col)
        //    screen.SetPixel(col,row,color);
    }
}

std::vector<Triangle>
GetTriangles(void)
{
    vtkPolyDataReader *rdr = vtkPolyDataReader::New();
    rdr->SetFileName("proj1d_geometry.vtk");
    cerr << "Reading" << endl;
    rdr->Update();
    cerr << "Done reading" << endl;
    if (rdr->GetOutput()->GetNumberOfCells() == 0)
    {
        cerr << "Unable to open file!!" << endl;
        exit(EXIT_FAILURE);
    }
    vtkPolyData *pd = rdr->GetOutput();
    int numTris = pd->GetNumberOfCells();
    vtkPoints *pts = pd->GetPoints();
    vtkCellArray *cells = pd->GetPolys();
    vtkFloatArray *var = (vtkFloatArray *) pd->GetPointData()->GetArray("hardyglobal");
    float *color_ptr = var->GetPointer(0);
    std::vector<Triangle> tris(numTris);
    vtkIdType npts;
    vtkIdType *ptIds;
    int idx;
    for (idx = 0, cells->InitTraversal() ; cells->GetNextCell(npts, ptIds) ; idx++)
    {
        if (npts != 3)
        {
            cerr << "Non-triangles!! ???" << endl;
            exit(EXIT_FAILURE);
        }
        tris[idx].X[0] = pts->GetPoint(ptIds[0])[0];
        tris[idx].X[1] = pts->GetPoint(ptIds[1])[0];
        tris[idx].X[2] = pts->GetPoint(ptIds[2])[0];
        tris[idx].Y[0] = pts->GetPoint(ptIds[0])[1];
        tris[idx].Y[1] = pts->GetPoint(ptIds[1])[1];
        tris[idx].Y[2] = pts->GetPoint(ptIds[2])[1];
        tris[idx].Z[0] = pts->GetPoint(ptIds[0])[2];
        tris[idx].Z[1] = pts->GetPoint(ptIds[1])[2];
        tris[idx].Z[2] = pts->GetPoint(ptIds[2])[2];
        // 1->2 interpolate between light blue, dark blue
        // 2->2.5 interpolate between dark blue, cyan
        // 2.5->3 interpolate between cyan, green
        // 3->3.5 interpolate between green, yellow
        // 3.5->4 interpolate between yellow, orange
        // 4->5 interpolate between orange, brick
        // 5->6 interpolate between brick, salmon
        double mins[7] = { 1, 2, 2.5, 3, 3.5, 4, 5 };
        double maxs[7] = { 2, 2.5, 3, 3.5, 4, 5, 6 };
        unsigned char RGB[8][3] = { { 71, 71, 219 }, 
                                    { 0, 0, 91 },
                                    { 0, 255, 255 },
                                    { 0, 128, 0 },
                                    { 255, 255, 0 },
                                    { 255, 96, 0 },
                                    { 107, 0, 0 },
                                    { 224, 76, 76 } 
                                  };
        for (int j = 0 ; j < 3 ; j++)
        {
            float val = color_ptr[ptIds[j]];
            int r;
            for (r = 0 ; r < 7 ; r++)
            {
                if (mins[r] <= val && val < maxs[r])
                    break;
            }
            if (r == 7)
            {
                cerr << "Could not interpolate color for " << val << endl;
                exit(EXIT_FAILURE);
            }
            double proportion = (val-mins[r]) / (maxs[r]-mins[r]);
            tris[idx].colors[j][0] = (RGB[r][0]+proportion*(RGB[r+1][0]-RGB[r][0]))/255.0;
            tris[idx].colors[j][1] = (RGB[r][1]+proportion*(RGB[r+1][1]-RGB[r][1]))/255.0;
            tris[idx].colors[j][2] = (RGB[r][2]+proportion*(RGB[r+1][2]-RGB[r][2]))/255.0;
        }
    }

    return tris;
}

int main()
{
   vtkImageData *image = NewImage(Nx, Ny);
   unsigned char *buffer = 
     (unsigned char *) image->GetScalarPointer(0,0,0);
   int npixels = Nx*Ny;
   // Initialize everything as black
   for (int i = 0 ; i < npixels*3 ; i++)
       buffer[i] = 0;

   std::vector<Triangle> triangles = GetTriangles();
   
   // Screen is a global object
   screen.buffer = buffer;

   // YOUR CODE GOES HERE TO DEPOSIT THE COLORS FROM TRIANGLES 
   // INTO PIXELS USING THE SCANLINE ALGORITHM
   for (int i = 0; i < triangles.size(); ++i)
       triangles[i].raster();
   WriteImage(image, "allTriangles");
}
