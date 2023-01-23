#include <iostream>
#include <climits>
using namespace std;

int find_gcd(int a,int b)
{
    int maxm=max(a,b),minm=min(a,b);
    while(maxm%minm!=0)
    {
        int rem=maxm%minm;
        maxm=minm;
        minm=rem;

    }
    return minm;

}

int main() {

    int t;

    cin>>t;

    while(t--)

    {
        int n;
        cin>>n;
        int arr[n];
        for(int i=0;i<n;i++)
        {
            cin>>arr[i];
        }
        int count[10001]={0};
        for(int i=0;i<n;i++)
        {
            count[arr[i]]++;
        }
        int gcd=count[arr[0]];
        for(int i=1;i<n;i++)
        {
            gcd=find_gcd(gcd,count[arr[i]]);
        }
        if(gcd==1)
        {
            cout<<"false"<<endl;
        }
        else
        {
            cout<<"true"<<endl;
        }
    }
    return 0;

}