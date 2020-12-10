基于xxx的一个简单复现。



## pursuit_evade部分

model_state是干嘛的？

定义的地方：
`self.model_state = np.zeros((4,) + map_matrix.shape, dtype=np.float32)`
(4,)+ 大概是空着后面的维数加到后面

0：map_state
1: pursuer_state
2: evader_state

个数少



# 

- 搞懂Animate是怎么画图的
- 学习一下map的机制 √
- 学习一下agent的机制 √
- 改一改跑一跑 

## 1 Animate
```python
    def animate(self, act_fn, nsteps, file_name, rate=1.5, verbose=False):
        """
            Save an animation to an mp4 file.
        """
        plt.figure(0)
        # run sim loop
        o = self.reset()
        file_path = "/".join(file_name.split("/")[0:-1])
        temp_name = join(file_path, "temp_0.png")
        # generate .pngs
        self.save_image(temp_name)
        removed = 0
        for i in range(nsteps):
            #a = act_fn(o)
            a = []
            for j in range(len(o)):
                a.append(act_fn[j](o[j]))
            o, r, done, info = self.step(a)
            temp_name = join(file_path, "temp_" + str(i + 1) + ".png")
            self.save_image(temp_name)
            removed += info['removed']
            if verbose:
                print(r, info)
            if done:
                break
        if verbose:
            print("Total removed:", removed)
        # use ffmpeg to create .pngs to .mp4 movie
        ffmpeg_cmd = "ffmpeg -framerate " + str(rate) + " -i " + join(
            file_path, "temp_%d.png") + " -c:v libx264 -pix_fmt yuv420p " + file_name
        call(ffmpeg_cmd.split())
        # clean-up by removing .pngs
        map(os.remove, glob.glob(join(file_path, "temp_*.png")))

    def save_image(self, file_name):
        plt.cla()
        plt.matshow(self.model_state[0].T, cmap=plt.get_cmap('Greys'), fignum=0)
        x, y = self.pursuer_layer.get_position(0)
        plt.plot(x, y, "r*", markersize=12)
        for i in range(self.pursuer_layer.n_agents()):
            x, y = self.pursuer_layer.get_position(i)
            plt.plot(x, y, "r*", markersize=12)
            if self.train_pursuit:
                ax = plt.gca()
                ofst = self.obs_range / 2.0
                ax.add_patch(
                    Rectangle((x - ofst, y - ofst), self.obs_range, self.obs_range, alpha=0.5,
                              facecolor="#FF9848"))
        for i in range(self.evader_layer.n_agents()):
            x, y = self.evader_layer.get_position(i)
            plt.plot(x, y, "b*", markersize=12)
            if not self.train_pursuit:
                ax = plt.gca()
                ofst = self.obs_range / 2.0
                ax.add_patch(
                    Rectangle((x - ofst, y - ofst), self.obs_range, self.obs_range, alpha=0.5,
                              facecolor="#009ACD"))

        xl, xh = -self.obs_offset - 1, self.xs + self.obs_offset + 1
        yl, yh = -self.obs_offset - 1, self.ys + self.obs_offset + 1
        plt.xlim([xl, xh])
        plt.ylim([yl, yh])
        plt.axis('off')
        plt.savefig(file_name, dpi=200)
```

## 2. pursuit_evade class
```python

```