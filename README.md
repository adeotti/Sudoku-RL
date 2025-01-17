## Usage 

```python
    class Test:
        def __init__(self, render:bool, episodes:int):
            
            self.timer = QTimer()
            self.render = render

            self.environmentClass = ENVI()
            self.terminated = False
            self.counter = 0
            self.episodes = episodes
        
        def stepComputing(self):
            self.environmentClass.reset()
            self.action = self.environmentClass.action_space.sample()
            observation, reward, terminated, truncated, info = self.environmentClass.step(self.action)
            return observation, reward, terminated
            
        def guiRender(self):
            if self.counter < self.episodes:
                self.environmentClass.render()
                self.stepComputing()
                self.counter += 1
            else:
                self.timer.stop()
                sys.exit()
        
        def run(self):
            if self.render:
                self.timer.timeout.connect(self.guiRender)
                self.timer.start(100)
                app.exec()
            else:
                while self.counter < self.episodes:
                    self.stepComputing()
                    print(self.action)
                    self.counter+=1
                    
    t = Test(render=True,episodes=12)
    t.run()
```