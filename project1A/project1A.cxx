#include <iostream>
#include <vtkDataSet.h>
#include <vtkImageData.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>

using std::cerr;
using std::endl;

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

void
colorStrip(int stripNumber, int rgb[3])
{
    // Set red (rgb[0])
    if (stripNumber/9 == 0)
    {
        rgb[0] = 0;
    }
    else if (stripNumber/9 == 1)
    {
        rgb[0] = 128;
    }
    else 
    {
        rgb[0] = 255;
    }
    
    // Set green (rgb[1])
    if ((stripNumber/3) % 3 == 0)
    {
        rgb[1] = 0;
    }
    else if ((stripNumber/3) % 3 == 1)
    {
        rgb[1] = 128;
    }
    else 
    {
        rgb[1] = 255;
    }

    // Set blue (rbg[2])
    if (stripNumber % 3 == 0)
    {
        rgb[2] = 0;
    }
    else if (stripNumber % 3 == 1)
    {
        rgb[2] = 128;
    }
    else 
    {
        rgb[2] = 255;
    }

}

/*
 * Write 27 horizontal strips (x), each of height 50 pixels
 */
void
writeStrips(unsigned char *buffer, unsigned int* size)
{
    int numPixels = size[0] * size[1];
    int rgb[3] = {0,0,0};
    int stripNumber = 0;
    // We can cheat because first block is black
    for ( int i = 1; i < numPixels; ++i)
    {
        if (i % (50 * size[0]) == 0)
            ++stripNumber;
        colorStrip(stripNumber, rgb);
        int index = 3*i;
        buffer[index+0] = rgb[0];
        buffer[index+1] = rgb[1];
        buffer[index+2] = rgb[2];
    }
}

int main()
{
   // Set image size (width, height)
   unsigned int size[2] = {1024,1350};
   vtkImageData *image = NewImage(size[0],size[1]);
   unsigned char *buffer = 
     (unsigned char *) image->GetScalarPointer(0,0,0);
   writeStrips(buffer, size);
   WriteImage(image, "proj1A");
}
