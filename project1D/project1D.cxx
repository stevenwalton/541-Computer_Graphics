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
      double          *zbuffer;
      int             width, height;
      // Functions
      void SetPixel(int c, int r, double z, double color[3]);
};


void
Screen::SetPixel(int c, int r, double z, double color[3])
{
    int pixel = ((r*width) + c);
    if (r < 0 || r >= height || c < 0 || c>= width)
        return;
    if(z < -1 || z > 0)
    {
        cerr << "ERROR: z=" << z << " which is out of bounds" << endl;
        abort();
    }
    //if(z < zbuffer[pixel])
    //    return;
    //zbuffer[pixel] = z;
    int index = pixel*3;
    for(int i = 0; i < 3; ++i)
        buffer[index+i] = ceil_441(color[i]*255);
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
        double          colors[3][3];
        double          ymin, ymax;
        int             ymin_index, ymax_index, ymid_index;
        // Functions
        void            raster();
        void            getMinMax();
        bool            isArbitrary();
        double          getIntercept(double x[3], double y[3]);
        void            splitTriangle();
        void            drawTriangle(double lX[3], double lY[3]);
        void            orientTriangleAndDraw();
};

void
Triangle::raster()
{
    getMinMax();
    if(isArbitrary())
        splitTriangle();
    else
        orientTriangleAndDraw();
}

void
Triangle::getMinMax()
{
    ymin = *std::min_element(Y,Y+3);
    ymax = *std::max_element(Y,Y+3);
    for(int i = 0; i < 3; ++i)
    {
        if(Y[i] != ymin && Y[i] != ymax)
            ymid_index = i;
        else if (Y[i] == ymin)
            ymin_index = i;
        else if (Y[i] == ymax)
            ymax_index = i;
    }
}

bool
Triangle::isArbitrary()
{
    for(int i = 0; i < 3; ++i)
    {
        if(Y[i] == Y[(i+1)%3])
            return false;
        //if(X[i] == X[(i+1)%3])
        //    return true;
    }
    return true;
}

void
Triangle::orientTriangleAndDraw()
{
    // Check if going down
    double x[3];
    double y[3];
    for(int i = 0; i < 3; ++i)
    {
        if(Y[i] == Y[(i+1)%3])
        {
            //if(Y[i] == maxY) // Going down triangle
            //{
            x[0] = X[i];
            y[0] = Y[i];
            x[1] = X[(i+1)%3];
            y[1] = Y[(i+1)%3];
            x[2] = X[(i+2)%3];
            y[2] = Y[(i+2)%3];
            //}
            //else if(Y[i] == minY) // Going up triangle
            //{
            //}
            //else
            //{
            //    cerr << "Something is seriously wrong" << endl;
            //    abort();
            //}
        }
    }
    if(!(y[2] == ymin || y[2] == ymax))
    {
        cerr << "y's aren't right" << endl;
        abort();
    }
    drawTriangle(x,y);
}

double
Triangle::getIntercept(double x[3], double y[3])
{
    double lmin = *std::min_element(y,y+3);
    double lmax = *std::max_element(y,y+3);
    int minIndex,midIndex, maxIndex;
    // Assumes there is a midpoint
    for(int i = 0; i < 3; ++i)
    {
        if (y[i] != lmin && y[i] != lmax)
            midIndex = i;
        else if(y[i] == lmin)
            minIndex = i;
        else if(y[i] == lmax)
            maxIndex = i;
    }
    if(x[maxIndex] == x[minIndex])
    {
        //cerr << "X's are the same" << endl;
        //cerr << x[maxIndex] << endl;
        //std::cin.ignore();
        return x[maxIndex];
    }
    double m = (y[maxIndex] - y[minIndex])/(x[maxIndex] - x[minIndex]);
    double b = y[maxIndex] - (m*x[maxIndex]);
    return ((y[midIndex] - b)/m);
}

void
Triangle::splitTriangle()
{
    /*
    cerr << "==================" << endl;
    cerr << "Original Triangle\n"
         << X[0] << " " << Y[0] << "\n"
         << X[1] << " " << Y[1] << "\n"
         << X[2] << " " << Y[2] 
         << endl;
    */
    double xintercept = getIntercept(X,Y);
    double lX[3] = {X[ymid_index], xintercept, X[ymin_index]};
    double lY[3] = {Y[ymid_index], Y[ymid_index], Y[ymin_index]};
    // Double check that is lower
    if(lY[2] > lY[1] || lY[2] > lY[0])
    {
        cerr << "Not a lower triangle" << endl;
        abort();
    }
    //cerr << endl;
    //cerr << "Lower Triangle" << endl;
    drawTriangle(lX,lY);
    double uX[3] = {X[ymid_index], xintercept, X[ymax_index]};
    double uY[3] = {Y[ymid_index], Y[ymid_index], Y[ymax_index]};
    // Double check that is upper
    if(uY[2] < uY[1] || uY[2] < uY[0])
    {
        cerr << "Not a upper triangle" << endl;
        abort();
    }
    //cerr << "Upper Triangle" << endl;
    drawTriangle(uX,uY);
    //cerr << "==================" << endl;
}

void
Triangle::drawTriangle(double x[3], double y[3])
{
    int lowerY = ceil_441(*std::min_element(y,y+3));
    int upperY = floor_441(*std::max_element(y,y+3));
    // Check that orientation is correct
    if(y[0] != y[1])
    {
        cerr << "Triangle isn't lower or upper" << endl;
        cerr 
             << x[0] << " " << y[0] << "\n"
             << x[1] << " " << y[1] << "\n"
             << x[2] << " " << y[2] 
             << endl;
        abort();
    }
    // Set x[0] on left side
    if(x[1] < x[0])
    {
        double x0 = x[0];
        x[0] = x[1];
        x[1] = x0;
    }
    /*
    cerr 
         << x[0] << " " << y[0] << "\n"
         << x[1] << " " << y[1] << "\n"
         << x[2] << " " << y[2] 
         << endl;
    cerr.flush();
    cerr << endl;
    */
    double m0 = 0;
    double m1 = 0;
    double b0, b1;
    if(x[0] != x[2]) // Not a right triangle with left vertical 
    {
        m0 = (y[2]-y[0])/(x[2]-x[0]);
        b0 = y[2] - (m0*x[2]);
    }
    if(x[1] != x[2]) // Not a right triangle with right vertical
    {
        m1 = (y[2]-y[1])/(x[2]-x[1]);
        b1 = y[2] - (m1*x[2]);
    }
    for(int row = lowerY; row <= upperY; ++row)
    {
        double leftX, rightX;
        if(m0 == 0)
            leftX = *std::min_element(x,x+3);
        else
            leftX = (row - b0)/m0;
        if(m1 == 0)
            rightX = *std::max_element(x,x+3);
        else
            rightX = (row - b1)/m1;
        leftX  = ceil_441(leftX);
        rightX = floor_441(rightX);
        for(int col = leftX; col <= rightX; ++col)
        {
            double color[3] = {1,1,1};
            screen.SetPixel(col,row,-1,color);
        }
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
   {
       //cerr << "Triangle: " << i << endl;
       triangles[i].raster();
   }
   WriteImage(image, "allTriangles");
}
