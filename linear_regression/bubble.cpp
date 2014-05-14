#include <iostream>

using namespace std;



void swap(int *x, int *y)

{

  int *temp;

  temp =x;

  x=y;

  y=temp;



}

void bubblesort(int data[], int size)

{

  for (int i=0; i<size; i++)

    {

      for (int j=0; j<size-1; j++)

      if (data[j-1] >  data[j])

        swap(data[j-1],data[j]);



    }

}

  int main()

{



  int size,*array,i;

  cout <<"Enter Size of Array"<<endl;

  cin>> size;



  array=new int[size];



  cout<< "Enter the Elements into array"<<endl;

  for(i=0;i<size;i++)

    cin>>array[i];





  cout<< "Unsorted Array"<<endl;

  for(i=0;i<size;i++)

    cout<<array[i]<<" ";



  bubblesort(array,size);

  cout<<"Sorted Array\n"<<endl;

  for(i=0;i<size;i++)

    cout<<array[i]<<" ";



  cout<<endl;



  return 0;





}
