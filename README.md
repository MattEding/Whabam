# Whabam!  

__Investigation__:  
Can a music artist be identified by a 30 second song segment using machine learning?

__Tools__:  
See requirements.txt for python libraries used. Also requires SQL database,
[ffmpeg](https://www.ffmpeg.org/) audio codec, and two music artist discographies to analyze.  

__Implementation__:  
Use [LibROSA](http://librosa.github.io/) to separate songs into 30 second segments and extract audio features.
Then use [scikit-learn](https://scikit-learn.org/) to make classification models.  

__Details__:  
Please reference [presentation](https://docs.google.com/presentation/d/1Zz4cyXY-baE-YiYz4doiyoJxmoUZdQvsKiWlfVguBV0/edit?usp=sharing) for details of findings.  
