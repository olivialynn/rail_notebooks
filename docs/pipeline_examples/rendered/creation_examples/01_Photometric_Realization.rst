Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``00_Quick_Start_in_Creation.ipynb``

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
```Photometric_Realization.ipynb`` <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization.ipynb>`__
in the ``interactive_examples/creation_examples/`` folder for a version
of this notebook in interactive mode.

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f664e696690>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398945  27.667538  26.723341  26.032640  25.178591  24.695961   
    1      2.285624  28.786999  27.476587  26.640171  26.259747  25.865673   
    2      1.495132  30.011347  29.789335  28.200388  26.014830  25.030172   
    3      0.842595  29.306242  28.721798  27.353014  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950443  25.687405  25.466604   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270803  26.371508  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042774  25.204935  24.825090   
    99997  2.023549  26.990147  26.714737  26.377949  26.250345  25.917370   
    99998  1.548204  26.367434  26.206886  26.087982  25.876932  25.715893   
    99999  1.739491  26.881981  26.773062  26.553120  26.319618  25.955982   
    
                   y     major     minor  
    0      23.994415  0.054451  0.036012  
    1      25.391064  0.132894  0.119264  
    2      24.304703  0.169769  0.153101  
    3      25.291103  0.206400  0.173595  
    4      25.096743  0.011070  0.007277  
    ...          ...       ...       ...  
    99995  24.737946  0.051465  0.038986  
    99996  24.224169  0.029701  0.027362  
    99997  25.613838  0.073353  0.055331  
    99998  25.274902  0.038116  0.021141  
    99999  25.699636  0.016744  0.011708  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model
    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398945</td>
          <td>26.957848</td>
          <td>0.532196</td>
          <td>26.925663</td>
          <td>0.198986</td>
          <td>26.004958</td>
          <td>0.079192</td>
          <td>25.128060</td>
          <td>0.059477</td>
          <td>24.705452</td>
          <td>0.078293</td>
          <td>23.826062</td>
          <td>0.081097</td>
          <td>0.054451</td>
          <td>0.036012</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.599965</td>
          <td>0.827183</td>
          <td>27.542235</td>
          <td>0.329546</td>
          <td>26.657224</td>
          <td>0.140013</td>
          <td>26.376063</td>
          <td>0.176759</td>
          <td>25.829195</td>
          <td>0.206776</td>
          <td>25.538563</td>
          <td>0.345043</td>
          <td>0.132894</td>
          <td>0.119264</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.727562</td>
          <td>0.381150</td>
          <td>28.332956</td>
          <td>0.539220</td>
          <td>25.873964</td>
          <td>0.114740</td>
          <td>25.225938</td>
          <td>0.123533</td>
          <td>24.400508</td>
          <td>0.133990</td>
          <td>0.169769</td>
          <td>0.153101</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.268447</td>
          <td>1.093374</td>
          <td>26.758371</td>
          <td>0.152733</td>
          <td>26.038601</td>
          <td>0.132366</td>
          <td>25.967214</td>
          <td>0.231968</td>
          <td>25.377513</td>
          <td>0.303539</td>
          <td>0.206400</td>
          <td>0.173595</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.104943</td>
          <td>0.275430</td>
          <td>26.008993</td>
          <td>0.090313</td>
          <td>26.000399</td>
          <td>0.078874</td>
          <td>25.769134</td>
          <td>0.104707</td>
          <td>25.686579</td>
          <td>0.183384</td>
          <td>25.119995</td>
          <td>0.246184</td>
          <td>0.011070</td>
          <td>0.007277</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>28.164254</td>
          <td>1.163587</td>
          <td>26.245117</td>
          <td>0.111039</td>
          <td>25.431638</td>
          <td>0.047641</td>
          <td>25.026323</td>
          <td>0.054342</td>
          <td>24.786860</td>
          <td>0.084122</td>
          <td>24.896799</td>
          <td>0.204512</td>
          <td>0.051465</td>
          <td>0.038986</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.726920</td>
          <td>0.168205</td>
          <td>26.065680</td>
          <td>0.083550</td>
          <td>25.232091</td>
          <td>0.065226</td>
          <td>24.861965</td>
          <td>0.089872</td>
          <td>24.263294</td>
          <td>0.118964</td>
          <td>0.029701</td>
          <td>0.027362</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.541562</td>
          <td>0.389366</td>
          <td>26.595549</td>
          <td>0.150347</td>
          <td>26.428230</td>
          <td>0.114809</td>
          <td>26.240953</td>
          <td>0.157537</td>
          <td>26.219984</td>
          <td>0.285324</td>
          <td>25.579974</td>
          <td>0.356467</td>
          <td>0.073353</td>
          <td>0.055331</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.356737</td>
          <td>0.336990</td>
          <td>26.138558</td>
          <td>0.101174</td>
          <td>26.021060</td>
          <td>0.080326</td>
          <td>25.938950</td>
          <td>0.121413</td>
          <td>25.585205</td>
          <td>0.168264</td>
          <td>27.981475</td>
          <td>1.644407</td>
          <td>0.038116</td>
          <td>0.021141</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.053887</td>
          <td>0.570384</td>
          <td>26.952389</td>
          <td>0.203498</td>
          <td>26.561285</td>
          <td>0.128875</td>
          <td>26.388073</td>
          <td>0.178569</td>
          <td>25.912953</td>
          <td>0.221751</td>
          <td>25.725879</td>
          <td>0.399288</td>
          <td>0.016744</td>
          <td>0.011708</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model_gaap


.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398945</td>
          <td>28.221012</td>
          <td>1.296578</td>
          <td>26.572806</td>
          <td>0.170512</td>
          <td>26.100736</td>
          <td>0.102014</td>
          <td>25.178750</td>
          <td>0.074299</td>
          <td>24.622607</td>
          <td>0.086211</td>
          <td>23.998812</td>
          <td>0.112328</td>
          <td>0.054451</td>
          <td>0.036012</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.619399</td>
          <td>0.414033</td>
          <td>26.514220</td>
          <td>0.152507</td>
          <td>26.296161</td>
          <td>0.204394</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.382144</td>
          <td>0.371285</td>
          <td>0.132894</td>
          <td>0.119264</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.891448</td>
          <td>0.477606</td>
          <td>25.759350</td>
          <td>0.133334</td>
          <td>24.974990</td>
          <td>0.126318</td>
          <td>24.135893</td>
          <td>0.136416</td>
          <td>0.169769</td>
          <td>0.153101</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>26.774925</td>
          <td>0.551054</td>
          <td>28.356103</td>
          <td>0.734703</td>
          <td>27.296464</td>
          <td>0.308944</td>
          <td>26.285209</td>
          <td>0.214478</td>
          <td>25.453108</td>
          <td>0.195396</td>
          <td>25.045186</td>
          <td>0.300075</td>
          <td>0.206400</td>
          <td>0.173595</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.199065</td>
          <td>0.330694</td>
          <td>26.239201</td>
          <td>0.127289</td>
          <td>26.154795</td>
          <td>0.106220</td>
          <td>25.771861</td>
          <td>0.124077</td>
          <td>25.673748</td>
          <td>0.211766</td>
          <td>24.744000</td>
          <td>0.211031</td>
          <td>0.011070</td>
          <td>0.007277</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.552279</td>
          <td>0.436825</td>
          <td>26.558079</td>
          <td>0.168367</td>
          <td>25.451698</td>
          <td>0.057528</td>
          <td>25.163341</td>
          <td>0.073283</td>
          <td>24.955668</td>
          <td>0.115387</td>
          <td>24.397561</td>
          <td>0.158488</td>
          <td>0.051465</td>
          <td>0.038986</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.531522</td>
          <td>0.163988</td>
          <td>25.858445</td>
          <td>0.082091</td>
          <td>25.321909</td>
          <td>0.083915</td>
          <td>24.819010</td>
          <td>0.101979</td>
          <td>24.277255</td>
          <td>0.142334</td>
          <td>0.029701</td>
          <td>0.027362</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>25.942537</td>
          <td>0.271772</td>
          <td>26.724177</td>
          <td>0.194972</td>
          <td>26.291238</td>
          <td>0.121284</td>
          <td>26.430482</td>
          <td>0.220443</td>
          <td>25.522355</td>
          <td>0.188971</td>
          <td>25.390019</td>
          <td>0.361091</td>
          <td>0.073353</td>
          <td>0.055331</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.001475</td>
          <td>0.282883</td>
          <td>26.314018</td>
          <td>0.136141</td>
          <td>26.353409</td>
          <td>0.126629</td>
          <td>26.176235</td>
          <td>0.176101</td>
          <td>25.634058</td>
          <td>0.205420</td>
          <td>24.979978</td>
          <td>0.257273</td>
          <td>0.038116</td>
          <td>0.021141</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.239674</td>
          <td>0.712498</td>
          <td>27.428204</td>
          <td>0.342645</td>
          <td>26.854646</td>
          <td>0.193921</td>
          <td>26.382575</td>
          <td>0.209023</td>
          <td>26.221293</td>
          <td>0.331095</td>
          <td>25.394544</td>
          <td>0.357938</td>
          <td>0.016744</td>
          <td>0.011708</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model_auto


.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398945</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.527930</td>
          <td>0.145291</td>
          <td>26.061690</td>
          <td>0.085631</td>
          <td>25.151145</td>
          <td>0.062539</td>
          <td>24.794914</td>
          <td>0.087133</td>
          <td>23.983355</td>
          <td>0.095889</td>
          <td>0.054451</td>
          <td>0.036012</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.935521</td>
          <td>0.582657</td>
          <td>27.513482</td>
          <td>0.371166</td>
          <td>26.721325</td>
          <td>0.175975</td>
          <td>26.288105</td>
          <td>0.196213</td>
          <td>25.831010</td>
          <td>0.245090</td>
          <td>24.835092</td>
          <td>0.231359</td>
          <td>0.132894</td>
          <td>0.119264</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.295198</td>
          <td>0.383106</td>
          <td>29.584942</td>
          <td>1.499967</td>
          <td>27.275585</td>
          <td>0.302065</td>
          <td>25.884437</td>
          <td>0.151819</td>
          <td>24.999686</td>
          <td>0.131872</td>
          <td>24.017638</td>
          <td>0.125913</td>
          <td>0.169769</td>
          <td>0.153101</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.252379</td>
          <td>0.316729</td>
          <td>26.550859</td>
          <td>0.284326</td>
          <td>25.285928</td>
          <td>0.180804</td>
          <td>25.008509</td>
          <td>0.309874</td>
          <td>0.206400</td>
          <td>0.173595</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.480294</td>
          <td>0.371564</td>
          <td>25.999319</td>
          <td>0.089642</td>
          <td>25.991042</td>
          <td>0.078319</td>
          <td>25.573593</td>
          <td>0.088309</td>
          <td>25.504297</td>
          <td>0.157217</td>
          <td>25.021377</td>
          <td>0.227176</td>
          <td>0.011070</td>
          <td>0.007277</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.082556</td>
          <td>0.275311</td>
          <td>26.251136</td>
          <td>0.114302</td>
          <td>25.567258</td>
          <td>0.055254</td>
          <td>25.088275</td>
          <td>0.059113</td>
          <td>24.867537</td>
          <td>0.092829</td>
          <td>24.920385</td>
          <td>0.214328</td>
          <td>0.051465</td>
          <td>0.038986</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.710653</td>
          <td>0.445972</td>
          <td>26.672206</td>
          <td>0.162031</td>
          <td>26.116371</td>
          <td>0.088330</td>
          <td>25.195144</td>
          <td>0.063863</td>
          <td>24.850023</td>
          <td>0.089913</td>
          <td>24.358959</td>
          <td>0.130719</td>
          <td>0.029701</td>
          <td>0.027362</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.393790</td>
          <td>0.743414</td>
          <td>26.507244</td>
          <td>0.145949</td>
          <td>26.131893</td>
          <td>0.093487</td>
          <td>26.348990</td>
          <td>0.182431</td>
          <td>26.479279</td>
          <td>0.368203</td>
          <td>27.602387</td>
          <td>1.406641</td>
          <td>0.073353</td>
          <td>0.055331</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.624560</td>
          <td>0.186588</td>
          <td>26.107667</td>
          <td>0.099565</td>
          <td>26.096624</td>
          <td>0.086960</td>
          <td>25.978760</td>
          <td>0.127345</td>
          <td>25.709428</td>
          <td>0.189253</td>
          <td>25.333696</td>
          <td>0.296588</td>
          <td>0.038116</td>
          <td>0.021141</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.165781</td>
          <td>0.618431</td>
          <td>26.984000</td>
          <td>0.209444</td>
          <td>26.615082</td>
          <td>0.135389</td>
          <td>26.084949</td>
          <td>0.138176</td>
          <td>26.086527</td>
          <td>0.256609</td>
          <td>27.000209</td>
          <td>0.969087</td>
          <td>0.016744</td>
          <td>0.011708</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_22_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_23_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
