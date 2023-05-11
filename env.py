from cProfile import label
from turtle import color
from matplotlib import scale
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Transform
import numpy as np



class custom_oneDEnv:

  def __init__(self, interval_size=1,simulation_dt=0.1, sf=2, tf=2, bias=0, ty="Deterministic"):
    super(custom_oneDEnv, self).__init__()

    # Size of the 1D-array
    self.interval_size = interval_size
    self.interval,self.dt=np.linspace(-self.interval_size,self.interval_size,1000,retstep=True)
    self.reset_pos=np.random.choice(self.interval) # init pos
    self.agent_pos = self.reset_pos # Init pos 
    self.agent_vel=0
    self.agent_acc=0
    self.simulation_dt=simulation_dt
    self.t = 0
    self.spatial_frequency = sf
    self.temporal_frequency = tf
    self.bias = bias
    self.ty = ty
    self.phase=1/2
  def set_pos(self,pos):
      self.agent_pos=pos

  def reset(self):
    """
    :return: (np.array) 
    """
    # Initialize the agent at the initial random pos
    self.agent_pos = self.reset_pos
    self.t = 0
    return np.array([self.agent_pos]).astype(np.float32)
  
  def get_output(self):
    #amp = np.sin(self.spatial_frequency*self.agent_pos)*np.sin(self.spatial_frequency*self.t) + self.bias
    amp = np.sin(self.spatial_frequency*self.agent_pos-self.phase)*np.sin(self.spatial_frequency*self.t) + self.bias
    if self.ty == "Deterministic":
        return amp
    elif self.ty == "Stochastic":
        return np.random.randn(1) * amp
    return 
  def step(self, action):
    self.t += self.simulation_dt
    if type(action)==float or  type(action)==int or type(np.ndarray):
        self.agent_acc=action
        self.agent_vel+=self.agent_acc*self.simulation_dt
        self.agent_pos+=self.agent_vel*self.simulation_dt
        
    else:
      raise ValueError("Received invalid action={} which is not part of the action space".format(action))

    self.agent_pos = np.clip(self.agent_pos, -self.interval_size, self.interval_size) # boundaries can't go further grid size
    self.agent_vel = np.clip(self.agent_vel, -self.interval_size, self.interval_size)
    self.agent_acc = np.clip(self.agent_acc, -self.interval_size, self.interval_size)
    done=None
    reward=None
    info = {}
    obs = self.get_output()
    return np.array([obs, self.agent_vel, self.agent_acc]).astype(np.float32) #, reward, done, info

  def render(self):
    """
      Real time simulation render
    """
      # create empty lists for the x and y data
    n_frames=int(1e3)
    x_limit=(0,n_frames)
    y_limit=(-self.interval_size,self.interval_size)
    x = []
    y = []

    # create the figure and axes objects
    fig, ax = plt.subplots()
    ax.set_xlim([x_limit[0],x_limit[1]])
    ax.set_ylim([y_limit[0],y_limit[1]])
   # function that draws each frame of the animation
    dt=self.simulation_dt
    def animate(dt):
        acc = np.random.uniform(-5,5) # grab random acc
        self.step(acc)
        x.append(self.agent_pos)
        y.append(self.get_output())

        #ax.clear()
        ax.plot(self.interval, y)
        ax.title.set_text("Sinus over time")
    ani = FuncAnimation(fig, animate, frames=n_frames, interval=50, repeat=False)
    #plt.plot(np.sin(self.spatial_frequency*self.interval)*np.sin(self.spatial_frequency*self.t) + self.bias)
    plt.show()


  def get_frame(self):
    sin_frame=np.sin(self.spatial_frequency*self.interval-self.phase)*np.sin(self.spatial_frequency*self.t) + self.bias
    sin_out=self.get_output()
    return sin_frame,sin_out,self.agent_pos,self.agent_vel,self.agent_acc,self.t

  def render_frames(self,frames,freq=1):
      fig, ax = plt.subplots()
      sclare_arrow=10
      for frame in frames:
        sin_frame,sin_out,pos,vel,acc=frame
        ax.set_xlim([-self.interval_size,self.interval_size])
        ax.set_ylim([-1.2,1.2])
        sin_=ax.plot(self.interval,sin_frame)
        o_line=ax.plot(self.interval,np.zeros_like(self.interval))
        #ax.quiver([pos,sin_out],vel,label="vel",color="b")
        origin=[pos,sin_out]
        acc_arrow=[acc,0]
        vec_=ax.quiver(*origin,*acc_arrow,label="acc",color="b",scale_units="x",scale=5)
        pos_=ax.scatter(pos,0,c='b',label="agent_pos")
        #ax.scatter(*origin,c='r',label="agent_sin_pos")
        ax.legend()
        ax.title.set_text("Sinus over time")
        plt.pause(1/freq)
        ax.clear()
      plt.show()
class Anim:
  def __init__(self,interval):
      self.interval=interval
  def init_frame(self,frame, ax):

      sin_frame,sin_out,pos,vel,acc, t=frame
      ax.set_xlim([self.interval[0],self.interval[-1]])
      ax.set_ylim([-1.2,1.2])
      self.sin_, =ax.plot(self.interval,sin_frame)
      self.o_line, =ax.plot(self.interval,np.zeros_like(self.interval))
      #ax.quiver([pos,sin_out],vel,label="vel",color="b")
      origin=[pos,sin_out]
      acc_arrow=[acc,0]
      self.vec_=ax.quiver(*origin,*acc_arrow,label="acc",color="b",scale_units="x",scale=5)
      #self.pos_=self.ax.scatter(pos,0,c='b',label="agent_pos")
      self.pos_=ax.scatter(*origin,c='r',label="agent_sin_pos")
      ax.title.set_text("Sinus over time")
      self.text=ax.text(0.2,0.9,"",Transform=ax.transAxes)
      ax.legend()
      return self.sin_,self.o_line,self.vec_,self.pos_,self.text

  def create_frame(self,frame):
      sin_frame,sin_out,pos,vel,acc,t=frame
      self.sin_.set_data(self.interval,sin_frame)
      origin=[pos,sin_out]
      acc_arrow=[acc,0]
      self.vec_.set_UVC(*acc_arrow)
      self.vec_.set_offsets(np.array(origin).T)
      self.pos_.set_offsets(np.array(origin).T)
      self.text.set_text("time= "+str(np.round(t,2)))
      return self.sin_,self.o_line,self.vec_,self.pos_,self.text
  

if __name__=="__main__":
  env=custom_oneDEnv(simulation_dt=0.1,sf=2)
  frames=[]
  for i in range(1000):
    acc = np.random.uniform(-5,5) # grab random acc
    env.step(acc)

    print("acc=",acc,"pos",env.get_frame()[2])
    frames+=[env.get_frame()]
  env.render_frames(frames,freq=2)