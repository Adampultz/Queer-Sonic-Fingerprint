<h1> Queer Sonic Fingerprint </h1>
<h2> Code for the sound installation <em> Queer Sonic Fingerprint</em> </h2>

This repository contains the code base for the sound installation <em> Queer Sonic Fingerprint </em> (QSF). QSF is a collaboration between Isabel Bredenbröker and Adam Pultz Melbye. The code is written by Adam Pultz.

This code is based around an evolutionary algorithm that takes frequency responses as its genome. The code introduces several novel evolutionary techniques, some of which are covered in the paper presentation Adam Pultz gave at the Speculative Sound Synthesis Symposium in Graz, September 2024: https://speculativesoundsynthesis.iem.sh/symposium/docs/proceedings/pultz/

Familiarity with SuperCollider is assumed.

<h3> Dependencies:</h3>

SuperCollider: https://supercollider.github.io/ <br>
sc3-Plugins: https://supercollider.github.io/sc3-plugins/ <br>
FluCoMa: https://learn.flucoma.org/installation/sc/ <br>
The Signal Box Quark (install this from SuperCollider itself) <br>

Additionally, you will need extensions written by me, which can be found here: <br>
https://github.com/Adampultz/Class-Extensions_Adam-Pultz/blob/main/Random%20Operators.sc

<h3> Preparing  audio</h3>

First, you will need to prepare two folders of audio files: the first folder is your impulse responses. You can use an IRs, but we have provided a folder of the IRs used in QSF here: https://www.dropbox.com/scl/fo/7te8zknifvlg1nsntaekt/AOGZJKOQOhFnDUUufUe02M4?rlkey=gujfk21rz9rhuz5evspk281mw&dl=0 <br>

Next you will need a folder of audio recordings for playback and filtering through the IRs. We recommend longer files, but you can technically use your IR folder as well. As these files take up a lot of space, we have not provided any for donwload.

In the SuperCollider file called `Main.scd"` locate "~objPath" (line 76) and make it reference your IR folder. `~museumPath` (line78) shuld references your audio recording folder. Currently, it is assumed that you're reading a folder of folders containing audio recordings organised into subfolders. If this is not the case, you need to change the function in "Functions.scd" called ~readFolderOfFolders. If you can't be bothered, you should be able to simply make a folder inside your audio folder which contains all of your recordings. The function automatically converts stereo files into mono.

<h3>How to run:</h3>

Check that you have chosen the right interface in `Main.scd` (top). <br>
Depending on how many speakers you use, you need to set ~numEnvironments in `Variables.scd` (default is four). <br>

Navigate to the "Run.scd" file and evaluate the line called `"Main.scd".loadRelative;` <br>

This will take a moment to load, based on the size of your files.

Once SC reports "Done", evaluate `~masterVol.set(1);` and the block of code in parenthesis under `"// Set amplitudes and thresholds"`

Finally, evaluate `~tasks[0].start;` to start the playback.

During playback, you can enable and disable processes, as well as adjust variables. Some can be accessed in the "Run.scd" file, while all are accessible from the "Variables.scd" file

<h3>Acknowledgments:</h3>

Queer Sonic Fingerprint premiered at Art Laboratory Berlin (https://artlaboratory-berlin.org/de/), to which we want to extend our deepest gratitude for supporting the project.

The development of the code base for Queer Sonic Fingerprint has been generously supported by Koda Kultur and Dansk Komponistforening. Some of the code was developed by a residency at Sound Art Lab (https://soundartlab.org/) and at a visiting guest artist fellowship at The Speculative Sound Synthesis Project at the the Institute of Electronic Music and Acoustics (IEM) at the University of Music and Performing Arts Graz (https://speculativesoundsynthesis.iem.sh/ ).
