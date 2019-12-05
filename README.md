# squat-counter
Using pose-net to count the number of squats done


# Example

![Demo](squat.gif)

## Credits for the model implementation [this repo](https://github.com/ildoonet/tf-pose-estimation)

- Impleted the use of posenet model and ran it locally also setting up environments

## Run
```py
python custom.py --model=mobilenet_thin --resize=432x368 --leg=right/left(choose) --vidlocation=something.mp4
```
