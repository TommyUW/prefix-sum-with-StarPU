#include<stdio.h>
#include<starpu.h>
#include<math.h>
#include<sys/time.h>
#include<xmmintrin.h>
#include<omp.h>
#define N 80000000
#define Num_Of_Task 4
#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define TAG(i,iter) ((starpu_tag_t) (((uint64_t)iter)<<32 | (i)) )
double time_value = 0.0;
double time_value2 = 0.0;
void prefix_cpu_func(void *buffers[], void *cl_arg)
{        
   int *t = cl_arg;
   int start = *t;      
   unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
   int *val = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
   int *result = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
   int end = n+start;

 //  struct timeval t1;
 //  struct timeval t2;

 //  gettimeofday(&t1,NULL);      
    int i;   
   // printf("start %d  end  %d\n",start, end);
    for( i = start;i<end;i++){
        if(i!=0){
            result[i] = result[i-1] + val[i];
            //printf("%d ",result[i]);    
        }       
    }
    /*       
        for(int i=start;i<start+20;i++){
            printf("result[%d] %d\n",i, result[i]);
        }
    */
 //   gettimeofday(&t2,NULL);
    //time_value += (t2.tv_sec-t1.tv_sec)+(t2.tv_usec-t1.tv_usec)/1000000.0;    
    //printf("time_value: %f\n", t2-);

}


struct starpu_codelet cl = 
{
    .where = STARPU_CPU,
    .cpu_funcs = {prefix_cpu_func},
    .cpu_funcs_name = {"prefix_cpu_func"},
    .nbuffers =2,
    .modes = {STARPU_RW,STARPU_RW} 
};


int main(int argc, char **argv){

    //int N[10] = {5120,6144,7168,8192,9216,10240,11264,12288,13312,14336};
   // int N;
    //while(N!=0){
    //printf("Enter the size of vector: \n");    
    //scanf("%d",&N);    
        int *vector = (int*)malloc(N*sizeof(int));
        memset(vector, 0, N*sizeof(int));
    
        int *res = (int*)malloc(N*sizeof(int));
        memset(res, 0, N*sizeof(int));
        for(int i=0;i<N;i++){
            vector[i] = i;
        }
        starpu_init(NULL);    
        int num_perTask = N/Num_Of_Task;
        int i = 0;

        while(i<Num_Of_Task)
        {
            starpu_data_handle_t vector_handle;
            starpu_data_handle_t res_handle;
            starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, num_perTask, sizeof(vector[0]));
            starpu_vector_data_register(&res_handle, STARPU_MAIN_RAM, (uintptr_t)res, num_perTask, sizeof(res[0]));

            struct starpu_task *task = starpu_task_create(); 
            int num = i*num_perTask;
            task->cl = &cl;
            task->handles[0] = vector_handle;
            task->handles[1] = res_handle;

            task->synchronous = 1;
            task->cl_arg = &num;
            task->cl_arg_size = sizeof(int); 
            task->tag_id = TAG(i,0);
            if(i!=0)
            starpu_tag_declare_deps(TAG(i,0),1,TAG(i-1,0));       
            struct timeval t1;    
            struct timeval t2;
            gettimeofday(&t1,NULL);      
            starpu_task_submit(task);      
            gettimeofday(&t2,NULL);   
            time_value += (t2.tv_sec-t1.tv_sec)+(t2.tv_usec-t1.tv_usec)/1000000.0;    
            i++;
        }   
    printf("Vector size: %d, Starpu time: %f\n",N,time_value);
    time_value = 0.0;
    //Sequencial
        int *result2 = (int*)malloc(N*sizeof(int));
        memset(result2, 0, N*sizeof(int));

        int *vector2 = (int*)malloc(N*sizeof(int));
        for(int i=0;i<N;i++){
            vector2[i] = i;
        }

        struct timeval t1;
        struct timeval t2;   
        gettimeofday(&t1,NULL);
        for( i = 0;i<N;i++){
            if(i!=0){
                result2[i] = result2[i-1] + vector2[i];
            //printf("%d ",result[i]);    
            }       
        }
        gettimeofday(&t2,NULL);
        time_value2 = (t2.tv_sec-t1.tv_sec)+(t2.tv_usec-t1.tv_usec)/1000000.0;        
        printf("Vector size: %d, serial time: %f\n",N,time_value2);
        //starpu_data_unregister(vector);
        starpu_shutdown();
        FPRINTF(stderr, "TEST DONE ...\n");
/*       
        for(int i=0;i<20;i++){
            printf("res[%d] %d\n",i, res[i]);
        }
        
        for(int i=0;i<20;i++){
            printf("result2[%d] %d\n",i, result2[i]);
        }
*/  
        //printf("vector[1] %d, vector[2] %d, vector[3] %d\n",vector[1],vector[2],vector[3]);

    for(int i = 0; i<N;i++){
        if(res[i]!=result2[i]){
            printf("False\n");
            printf("i: %d\n",i);
            break;
        }   
    }
    //}
    
}
