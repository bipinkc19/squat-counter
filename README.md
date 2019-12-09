# squat-counter
Using pose-net to count the number of squats done.


# Example

![Demo](squat.gif)
<br>
![Demo](press.gif)
## Credits for the model implementation [this repo](https://github.com/ildoonet/tf-pose-estimation)

- Implemented the use of posenet model and ran it locally setting up the environment.

## Run
```bash
➜ python squat_counter.py --model=mobilenet_thin --resize=432x368 --leg=right/left(choose) --vidlocation=something.mp4
```

```bash
➜ python bench_press_counter.py --model=mobilenet_thin --resize=432x368 --leg=right/left(choose) --vidlocation=something.mp4
```
