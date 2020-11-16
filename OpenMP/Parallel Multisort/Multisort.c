#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

#define MAX_SIZE 40000

//Δημιουργία πίνακα αριθμών με τυχαία σειρά
void generate_list(int *listPtr, int numOfElements){

  time_t t;
  srand((unsigned) time(&t));

  int temp,j;

  for (int i = 0; i < numOfElements; i++)
    listPtr[i]=i;

  for(int i = 0; i < numOfElements; i++){
    j = rand()%numOfElements;
    temp = listPtr[i];
    listPtr[i] = listPtr[j];
    listPtr[j] = temp;
  }

}

//Ένωση πινάκων σε έναν ενιαίο και ταξινομημένο πίνακα
void marge(int *firstQuarterStart, int *firstQuarterEnd, int *secQuarterStart, int *secQuarterEnd, int *tempListPtr){

  //Για όσο υπάρχουν στοιχεία στα δυο κομμάτια πίνακα, σύγκρινε και ταξινόμησε
  while(firstQuarterStart <= firstQuarterEnd && secQuarterStart <= secQuarterEnd) {
    if (firstQuarterStart[0] < secQuarterStart[0]) {
      tempListPtr[0] = firstQuarterStart[0];
      tempListPtr++;
      firstQuarterStart++;
    }else{
      tempListPtr[0] = secQuarterStart[0];
      tempListPtr++;
      secQuarterStart++;
    }
  }

  while (firstQuarterStart <= firstQuarterEnd) {
    tempListPtr[0] = firstQuarterStart[0];
    tempListPtr++;
    firstQuarterStart++;
  }

  while (secQuarterStart <= secQuarterEnd) {
    tempListPtr[0] = secQuarterStart[0];
    tempListPtr++;
    secQuarterStart++;
  }
}

//Αντιμετάθεση στοιχείων
void swap(int *element1, int *element2){

  int temp[1];

  temp[0] = element1[0];
  element1[0] = element2[0];
  element2[0] = temp[0];

}

//Αλγόριθμος ταξινόμησης. Χρησιμοποιεί έναν αριθμό σαν βάση για να συγκρίνει και να ταξινομήσει τα
//στοιχεία του πίνακα. Ο αριθμός αυτός μπορεί να αλλάξει στην πορεία.
void quicksort(int *listPtr,int numOfElements, int fElement, int lElement){

  int pivot=numOfElements/2,leftPtr,rightPtr;

  if (fElement < lElement) {
    pivot = fElement;
    leftPtr=fElement;
    rightPtr=lElement;

    while (leftPtr < rightPtr) {

      while (listPtr[leftPtr] <= listPtr[pivot] && leftPtr < lElement)
        leftPtr++;

      while (listPtr[rightPtr] > listPtr[pivot])
        rightPtr--;

      if (leftPtr < rightPtr)
        swap(&listPtr[leftPtr], &listPtr[rightPtr]);

    }

    swap(&listPtr[pivot],&listPtr[rightPtr]);
    quicksort(listPtr,numOfElements,fElement,rightPtr-1);
    quicksort(listPtr,numOfElements,rightPtr+1,lElement);
  }

}

//Σπάει τον πίνακα σε 3 ίσα κομμάτια, το 4ο αποτελείται από το περισσευούμενο κομμάτι του πίνακα,
//και το κάθε κομμάτι σε άλλα 4 μέχρι να μην γίνεται να σπάσει άλλο. Τα κάνει sort με χρήση της
//quicksort και τα ξαναενώνει ανά δυο.
void multisort(int *listPtr, int numOfElements, int *tempListPtr){

  int quarter,half,rest;

  quarter = numOfElements/4;
  half = 2*quarter;
  rest = numOfElements-3*quarter;

  if(numOfElements < 4){
    quicksort(listPtr,numOfElements,0,numOfElements-1);
    return;
  }

  #pragma omp task firstprivate(listPtr, numOfElements, tempListPtr)
  multisort(listPtr,quarter,tempListPtr);

  #pragma omp task firstprivate(listPtr, numOfElements, tempListPtr)
  multisort(listPtr+quarter,quarter,tempListPtr+quarter);

  #pragma omp task firstprivate(listPtr, numOfElements, tempListPtr)
  multisort(listPtr+half,quarter,tempListPtr+half);

  #pragma omp task firstprivate(listPtr, numOfElements, tempListPtr)
  multisort(listPtr+3*quarter,rest,tempListPtr+3*quarter);

  #pragma omp taskwait

  #pragma omp task
  marge(listPtr,listPtr+quarter-1,listPtr+quarter,listPtr+half-1,tempListPtr);

  #pragma omp task
  marge(listPtr+half,listPtr+3*quarter-1,listPtr+3*quarter,listPtr+numOfElements-1,tempListPtr+half);

  #pragma omp taskwait
  marge(tempListPtr,tempListPtr+half-1,tempListPtr+half,tempListPtr+numOfElements-1,listPtr);
}

void print_list(int *listPtr, int numOfElements){

  for (int i = 0; i < numOfElements; i++){
    if (i == 0) {
      printf("|");
    }

    if (i+1 == numOfElements) {
      printf("%d|\n\n",listPtr[i]);
    }else
      printf("%d ",listPtr[i]);
  }

}

int main(int argc, char *argv[]){

  omp_set_num_threads(4);

  int list[MAX_SIZE],numOfElements=10,tempList[MAX_SIZE];

  generate_list(list,numOfElements);

  printf("\nGenerated list:\n\n");
  print_list(list,numOfElements);

  double timer_start = omp_get_wtime();

  //Ορισμός του παράλληλου τμήματος
  #pragma omp parallel
  {
    //Η multisort θα τρέξει μόνο από ένα νήμα, τα υπόλοιπα θα εκτελέσουν τα tasks
    //που θα δημιουργηθούν κατά την κλήση της multisort.
    #pragma omp single
    multisort(list,numOfElements,tempList);
  }

  double timer_stop = omp_get_wtime();

  printf("\nSorted list:\n\n");
  print_list(list,numOfElements);

  printf("Elapsed time: %f\n\n",timer_stop - timer_start);

  return 0;
}
