class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad
        
    def forward(self, x):
        n, c, h, w=x.shape
        out_h=int(1+(h-self.pool_h)/self.stride)
        out_w=int(1+(w-self.pool_w)/self.stride)
        
        col=im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)  #전개
        col=col.reshape(-1, self.pool_h*self.pool_w)
        
        out=np.max(col, axis=1) #최댓값
        
        out=out.reshape(n, out_h, out_w, c).transpose(0, 3, 1, 2)
        
        return out
