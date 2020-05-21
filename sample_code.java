public static void sort(int[] arr){
    int n = arr.length;
    int temp = 0;
    for(int i=0; i < n; i++){
        for(int j=1; j < (n-i); j++){
            if(arr[j-1] > arr[j]){
                temp = arr[j-1];
                arr[j-1] = arr[j];
                arr[j] = temp;
            }
        }
    }
}

private String toHex(byte[] data){
  char[] chars=new char[data.length * 2];
  for (int i=0; i < data.length; i++) {
    chars[i * 2]=HEX_DIGITS[(data[i] >> 4) & 0xf];
    chars[i * 2 + 1]=HEX_DIGITS[data[i] & 0xf];
  }
  return new String(chars);
}

public static void e(String msg){
  if (LOG_ENABLE) {
    Log.e(TAG,buildMsg(msg));
  }
}

public static int max(final int a,final int b){
  return (a <= b) ? b : a;
}

public boolean add(Node n){
  int num=n.getNumber();
  if (!get(num)) {
    set(num);
    return true;
  }
 else   return false;
}