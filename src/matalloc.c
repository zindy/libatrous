/*
 * Copyright (c) 2013 Egor Zindy
 *
 * matalloc is free software: you can redistribute it and/or modify  
 * it under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>

#include "matalloc.h"

// Define
#define NEW(type) ((type *)calloc(1,sizeof(type)))    //allocation shortcut

/* 
   notations inspired by Meschach  https://github.com/yageek/Meschach
   ten->l, ten->m, ten->n is the size of a tensor (l planes, m rows, n cols) 
   ten->dsize is the element size

   mat->m, mat->n is the size of the array (m rows, n cols)
   mat->dsize is the element size

   vec->dim is the size of a vector
   vec->dsize is the element size

   CAREFUL! Need to cast te / me / ve first
   double ***array = (double ***)ten->te
   array[k][j][i] kth plane, jth line, ith col.

   ten->te[k][j][i] kth plane, jth line, ith col.
   mat->me[j][i] jth line, ith col.
   vec->ve[i] ith element of a vector

   ten=t_get(4,5,10) to define a 4 plane, 5 rows, 10 cols tensor
   mat=m_get(5,10) to define a 5 rows, 10 cols matrix
   vec=v_get(10) to define a 10 element vector.
   t_free(ten) m_free(mat) v_free(vec) to free the memory used by the structures.
*/

/********************************************************************
 allocation helper functions
 http://stackoverflow.com/questions/1919183/how-to-allocate-and-free-aligned-memory-in-c
********************************************************************/
#define ALIGN 64
void *aligned_calloc(size_t size) {
    void *mem = calloc(size+ALIGN+sizeof(void*),1);
    void **ptr = NULL;
    
    if (mem != NULL) {
        ptr = (void**)((uintptr_t)(mem+ALIGN+sizeof(void*)) & ~(uintptr_t)(ALIGN-1));
        ptr[-1] = mem;
    }
    return ptr;
}

void *aligned_malloc(size_t size) {
    void *mem = malloc(size+ALIGN+sizeof(void*));
    void **ptr = NULL;

    if (mem != NULL) {
        ptr = (void**)((uintptr_t)(mem+ALIGN+sizeof(void*)) & ~(uintptr_t)(ALIGN-1));
        ptr[-1] = mem;
    }
    return ptr;
}

void aligned_free(void *ptr) {
    if (ptr != NULL) free(((void**)ptr)[-1]);
}

/********************************************************************
 tensor functions
********************************************************************/
tensor *t_wrap2(void **data, int l,int m, int n, size_t dsize)
{
    //here with a discontiguous array along l data is a list of pointers to 2-D arrays.
    int i,j,k;
    char *ptr;
    int stride;

    tensor *ten = NEW(tensor);
    if (data == NULL || ten == NULL) {
        errno = ENOMEM; goto end;
    }

    ten->l=l;
    ten->m=m;
    ten->n=n;
    ten->dsize = dsize;
    //ten->is_managed = 0; // should already be 0 because calloc

    //we have no data... default case is_managed = 0
    ten->data=NULL;
    ten->me=(void **)malloc(l*m*sizeof(void *));
    ten->te=(void ***)malloc(l*sizeof(void *));

    if (ten->me == NULL || ten->te == NULL) {
        errno = ENOMEM; goto end;
    }

    //that's for me. Each array pointed to by data[i] is contiguous.
    k=0;
    stride = dsize * n;

    for (i=0;i<l;i++)
    {
        ptr = (char *)(data[i]);
        for (j=0;j<m;j++)
        {
            ten->me[k++] = (void *)ptr;
            ptr += stride;
        }
    }

    //that's for te
    k=0;
    for (i=0;i<l;i++)
    {
        ten->te[i]=ten->me+k;
        k+=m;
    }

end:
    if (errno) t_free(ten);
    return ten;
}

tensor *t_get2(int l,int m, int n, size_t dsize)
{
    //maybe we don't need a completely contiguous block of memory...
    void **data = NULL;
    tensor *ten = NULL;
    int i=0;

    data=(void **)calloc(l, sizeof(void *));
    if (data == NULL) {
        errno = ENOMEM; goto end;
    }

    for (i=0;i<l;i++) {
        data[i] = aligned_calloc(m*n*dsize);
        if (data[i] == NULL) {
            errno = ENOMEM;
            break;
        }
    }
    if (errno) goto end;
    ten = t_wrap2(data, l, m, n, dsize);

end:
    if (errno) {
        t_free(ten);
    } else {
        ten->is_managed = 2; //managed but non contiguous.
    }
    free(data);

    return ten;
}

tensor *t_wrap(void *data, int l,int m, int n, size_t dsize)
{
    int i, j, k;
    char *ptr;
    int stride;

    tensor *ten = NEW(tensor);
    if (data == NULL || ten == NULL) {
        errno = ENOMEM; goto end;
    }

    ten->l = l;
    ten->m = m;
    ten->n = n;
    ten->dsize = dsize;
    //ten->is_managed = 0; // should already be 0 because calloc

    ten->data=data;
    ten->me=(void **)malloc(l*m*sizeof(void *));
    ten->te=(void ***)malloc(l*sizeof(void *));

    if (ten->me == NULL || ten->te == NULL) {
        errno = ENOMEM; goto end;
    }

    //that's for me
    k=0;
    ptr = (char *)data;
    stride = dsize * n;

    for (i=0;i<l;i++)
    {
        for (j=0;j<m;j++)
        {
            ten->me[k++] = (void *)ptr;
            ptr += stride;
        }
    }

    //that's for te
    k=0;
    for (i=0;i<l;i++)
    {
        ten->te[i]=ten->me+k;
        k+=m;
    }

end:
    if (errno) t_free(ten);
    return ten;
}

tensor *t_get(int l,int m, int n, size_t dsize)
{
    void *data = NULL;
    tensor *ten = NULL;

    data = aligned_calloc(l*m*n*dsize);
    if (data == NULL) {
        errno = ENOMEM; goto end;
    }

    ten = t_wrap(data, l, m, n, dsize);
    if (ten != NULL) ten->is_managed = 1;

end:
    return ten;
}

void t_free(tensor *ten)
{
    int i;
    void *plane = NULL;

    if (ten==NULL) return;

    //if just a wrapper, don't deallocate the data
    //if non-contiguous, then need to free each plane separately.
    if (ten->is_managed == 1 && ten->data != NULL) aligned_free(ten->data);
    else if (ten->is_managed == 2 && ten->te != NULL) {
        for (i=0;i<ten->l;i++) {
            plane = ten->te[i][0];
            if (plane != NULL) aligned_free(plane);
        }
    }

    if (ten->me != NULL) free(ten->me);
    if (ten->te != NULL) free(ten->te);
    free(ten);
}


/********************************************************************
 matrix functions
********************************************************************/
matrix *m_wrap(void *data, int m, int n, size_t dsize)
{
    int i;
    char *ptr;
    int stride;

    matrix *mat = NEW(matrix);
    if (data == NULL || mat == NULL) {
        errno = ENOMEM; goto end;
    }

    mat->m=m; mat->n=n;

    mat->data=data;
    mat->me=(void **)malloc(m*sizeof(void *));

    if (mat->me == NULL) {
        errno = ENOMEM; goto end;
    }

    //stride is image line size bytes
    stride = dsize * n;
    ptr = (char *)data;

    for (i=0;i<m;i++)
    {
        mat->me[i] = (void *)ptr;
        ptr += stride;
    }

end:
    if (errno) m_free(mat);
    return mat;
}

matrix *m_get(int m, int n, size_t dsize)
{
    void *data = NULL;
    matrix *mat = NULL;

    data = aligned_calloc(m*n*dsize);
    if (data == NULL) {
        errno = ENOMEM; goto end;
    }

    mat = m_wrap(data,m,n,dsize);
    mat->is_managed = 1;

end:
    return mat;
}

void m_free(matrix *mat)
{
    if (mat==NULL) return;

    //if just a wrapper, don't deallocate the data
    if (mat->is_managed == 1 && mat->data != NULL) aligned_free(mat->data);

    if (mat->me != NULL) free(mat->me);
    free(mat);
}

/********************************************************************
 vector functions
********************************************************************/
vector *v_wrap(void *data, int dim, size_t dsize)
{
    vector *vec = NEW(vector);
    if (data == NULL || vec == NULL) {
        errno = ENOMEM; goto end;
    }

    vec->ve=data; vec->dim=dim;

end:
    if (errno) v_free(vec);
    return vec;
}

vector *v_get(int dim, size_t dsize)
{
    vector *vec = NULL;
    void *data = NULL;

    data = aligned_calloc(dim*dsize);
    if (data == NULL) {
        errno = ENOMEM; goto end;
    }

    vec=v_wrap(data,dim,dsize);

end:
    return vec;
}

void v_free(vector *vec)
{
    if (vec==NULL) return;

    if (vec->is_managed == 1 && vec->ve != NULL) aligned_free(vec->ve);
    free(vec);
}

/********************************************************************
 tensor functions
********************************************************************/
tensor *t_copy(tensor *ten1, tensor *tenOut) {
    int nx, ny, nz;
    int i;
    size_t dsize;
    void *plane1 = NULL, *planeOut = NULL;

    if (ten1 == NULL) {
        errno = ENOMEM; goto end;
    }

    nz = ten1->l; ny = ten1->m; nx = ten1->n; dsize = ten1->dsize;

    if (tenOut == NULL)
    {
        tenOut = t_get(nz,ny,nx,dsize);
        if (tenOut == NULL) {
            errno = ENOMEM; goto end;
        }
    }
    else
    {
        if (nz != tenOut->l || ny != tenOut->m || nx != tenOut->n || dsize != tenOut->dsize) {
            errno = EACCES; goto end;
        }
    }

    if (ten1->is_managed != 2 && tenOut->is_managed !=2) {
        memcpy(tenOut->data, ten1->data, nz*ny*nx*dsize);
    } else {
        for (i=0; i<nz; i++) {
            plane1 = ten1->te[i][0];
            planeOut = tenOut->te[i][0];

            if (plane1 != NULL && planeOut != NULL)
                memcpy(planeOut, plane1, ny*nx*dsize);
        }
    }

end:
    if (errno) t_free(tenOut);
    return tenOut;
}

void t_zero(tensor *ten1) {
    int nx,ny,nz;
    int i;
    size_t dsize;
    void *plane = NULL;

    if (ten1 == NULL) {
        errno = ENOMEM; goto end;
    }

    nz = ten1->l; ny = ten1->m; nx = ten1->n; dsize = ten1->dsize;
    
    //special case, non contiguous...
    if (ten1->is_managed == 2 && ten1->te != NULL) {
        for (i=0; i<nz; i++) {
            plane = ten1->te[i][0];
            if (plane != NULL)
                memset(plane, 0, ny*nx*dsize);
        }
    } else {
        memset(ten1->data, 0, nz*ny*nx*dsize);
    }

end:
    return;
}

void t_info(tensor *ten1) {
    printf("t->l=%d t->m=%d t->n=%d t->dsize=%d\n",ten1->l,ten1->m,ten1->n,(int)ten1->dsize);
}

/********************************************************************
 matrix functions
********************************************************************/
matrix *m_copy(matrix *mat1, matrix *matOut) {
    int nx, ny;
    size_t dsize;

    if (mat1 == NULL) {
        errno = ENOMEM; goto end;
    }

    ny = mat1->m; nx = mat1->n; dsize = mat1->dsize;

    if (matOut == NULL)
    {
        matOut = m_get(ny,nx,dsize);
        if (matOut == NULL) {
            errno = ENOMEM; goto end;
        }
    }
    else
    {
        if (ny != matOut->m || nx != matOut->n || dsize != matOut->dsize) {
            errno = EACCES; goto end;
        }
    }
    memcpy(matOut->data, mat1->data, ny*nx*dsize);

end:
    if (errno) m_free(matOut);
    return matOut;
}

void m_zero(matrix *mat1) {
    int nx,ny;
    size_t dsize;

    if (mat1 == NULL) {
        errno = ENOMEM; goto end;
    }

    ny = mat1->m; nx = mat1->n; dsize = mat1->dsize;
    memset(mat1->data, 0, ny*nx*dsize);

end:
    return;
}

void m_info(matrix *mat1) {
    printf("m->m=%d m->n=%d m->dsize=%d\n",mat1->m,mat1->n,(int)mat1->dsize);
}

/********************************************************************
 vector functions
********************************************************************/
vector *v_copy(vector *vec1, vector *vecOut) {
    int dim;
    size_t dsize;

    if (vec1 == NULL) {
        errno = ENOMEM; goto end;
    }

    dim = vec1->dim; dsize = vec1->dsize;

    if (vecOut == NULL)
    {
        vecOut = v_get(dim,dsize);
        if (vecOut == NULL) {
            errno = ENOMEM; goto end;
        }
    }
    else
    {
        if (dim != vecOut->dim || dsize != vecOut->dsize) {
            errno = EACCES; goto end;
        }
    }
    memcpy(vecOut->ve, vec1->ve, dim*dsize);

end:
    if (errno) v_free(vecOut);
    return vecOut;
}

void v_zero(vector *vec1) {
    int dim;
    size_t dsize;

    if (vec1 == NULL) {
        errno = ENOMEM; goto end;
    }

    dim = vec1->dim; dsize = vec1->dsize;
    memset(vec1->ve, 0, dim*dsize);

end:
    return;
}

void v_info(vector *vec1) {
    printf("v->dim=%d v->dsize=%d\n",vec1->dim,(int)vec1->dsize);
}

