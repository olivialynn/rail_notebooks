Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

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


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f5561a0eb30>



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
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.086743  0.067246  
    1      25.391064  0.000121  0.000083  
    2      24.304707  0.008646  0.005456  
    3      25.291103  0.120120  0.081663  
    4      25.096743  0.011165  0.007867  
    ...          ...       ...       ...  
    99995  24.737946  0.030979  0.016659  
    99996  24.224169  0.037964  0.026094  
    99997  25.613836  0.052501  0.027499  
    99998  25.274899  0.076657  0.068339  
    99999  25.699642  0.033107  0.020914  
    
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
          <td>1.398944</td>
          <td>28.260908</td>
          <td>1.228083</td>
          <td>26.516494</td>
          <td>0.140473</td>
          <td>26.097241</td>
          <td>0.085906</td>
          <td>25.157961</td>
          <td>0.061076</td>
          <td>24.802176</td>
          <td>0.085265</td>
          <td>24.098824</td>
          <td>0.103064</td>
          <td>0.086743</td>
          <td>0.067246</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.137084</td>
          <td>0.237310</td>
          <td>27.090649</td>
          <td>0.202482</td>
          <td>26.453519</td>
          <td>0.188734</td>
          <td>26.353936</td>
          <td>0.317752</td>
          <td>25.242501</td>
          <td>0.272152</td>
          <td>0.000121</td>
          <td>0.000083</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.301710</td>
          <td>0.584655</td>
          <td>29.869000</td>
          <td>1.404652</td>
          <td>26.150813</td>
          <td>0.145815</td>
          <td>25.012812</td>
          <td>0.102589</td>
          <td>24.266007</td>
          <td>0.119245</td>
          <td>0.008646</td>
          <td>0.005456</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.736162</td>
          <td>0.786920</td>
          <td>27.438761</td>
          <td>0.270062</td>
          <td>26.346454</td>
          <td>0.172369</td>
          <td>25.516308</td>
          <td>0.158656</td>
          <td>26.064732</td>
          <td>0.515185</td>
          <td>0.120120</td>
          <td>0.081663</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.177513</td>
          <td>0.292075</td>
          <td>26.123375</td>
          <td>0.099839</td>
          <td>25.997928</td>
          <td>0.078702</td>
          <td>25.773770</td>
          <td>0.105132</td>
          <td>25.693805</td>
          <td>0.184508</td>
          <td>25.024074</td>
          <td>0.227419</td>
          <td>0.011165</td>
          <td>0.007867</td>
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
          <td>29.350358</td>
          <td>2.069121</td>
          <td>26.417770</td>
          <td>0.128999</td>
          <td>25.443456</td>
          <td>0.048143</td>
          <td>25.138212</td>
          <td>0.060016</td>
          <td>24.627557</td>
          <td>0.073085</td>
          <td>24.938783</td>
          <td>0.211826</td>
          <td>0.030979</td>
          <td>0.016659</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.819941</td>
          <td>0.480875</td>
          <td>26.748230</td>
          <td>0.171281</td>
          <td>26.117493</td>
          <td>0.087452</td>
          <td>25.161662</td>
          <td>0.061277</td>
          <td>24.754363</td>
          <td>0.081746</td>
          <td>24.126075</td>
          <td>0.105550</td>
          <td>0.037964</td>
          <td>0.026094</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.267053</td>
          <td>0.313820</td>
          <td>26.935711</td>
          <td>0.200672</td>
          <td>26.459748</td>
          <td>0.118003</td>
          <td>26.395068</td>
          <td>0.179630</td>
          <td>26.498119</td>
          <td>0.356158</td>
          <td>25.816259</td>
          <td>0.427893</td>
          <td>0.052501</td>
          <td>0.027499</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.453927</td>
          <td>0.363735</td>
          <td>26.262741</td>
          <td>0.112756</td>
          <td>25.996922</td>
          <td>0.078633</td>
          <td>25.736836</td>
          <td>0.101789</td>
          <td>25.534373</td>
          <td>0.161125</td>
          <td>24.992840</td>
          <td>0.221592</td>
          <td>0.076657</td>
          <td>0.068339</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.336376</td>
          <td>0.331605</td>
          <td>26.584404</td>
          <td>0.148917</td>
          <td>26.579937</td>
          <td>0.130972</td>
          <td>26.246645</td>
          <td>0.158306</td>
          <td>26.036986</td>
          <td>0.245727</td>
          <td>25.847940</td>
          <td>0.438309</td>
          <td>0.033107</td>
          <td>0.020914</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



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
          <td>1.398944</td>
          <td>27.314158</td>
          <td>0.757354</td>
          <td>26.492241</td>
          <td>0.161013</td>
          <td>26.075794</td>
          <td>0.101102</td>
          <td>25.130154</td>
          <td>0.072137</td>
          <td>24.657137</td>
          <td>0.090024</td>
          <td>23.998917</td>
          <td>0.113822</td>
          <td>0.086743</td>
          <td>0.067246</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.687116</td>
          <td>0.418712</td>
          <td>26.456472</td>
          <td>0.137991</td>
          <td>26.085806</td>
          <td>0.162529</td>
          <td>26.227969</td>
          <td>0.332639</td>
          <td>26.185447</td>
          <td>0.643101</td>
          <td>0.000121</td>
          <td>0.000083</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.481137</td>
          <td>1.346058</td>
          <td>27.683953</td>
          <td>0.380080</td>
          <td>26.334799</td>
          <td>0.200713</td>
          <td>25.060697</td>
          <td>0.125555</td>
          <td>24.258236</td>
          <td>0.139655</td>
          <td>0.008646</td>
          <td>0.005456</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.206981</td>
          <td>1.303426</td>
          <td>28.777104</td>
          <td>0.916587</td>
          <td>27.335038</td>
          <td>0.297450</td>
          <td>26.008698</td>
          <td>0.157541</td>
          <td>25.853281</td>
          <td>0.253754</td>
          <td>24.938805</td>
          <td>0.256301</td>
          <td>0.120120</td>
          <td>0.081663</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.856758</td>
          <td>0.250922</td>
          <td>26.113568</td>
          <td>0.114144</td>
          <td>25.896120</td>
          <td>0.084652</td>
          <td>25.716788</td>
          <td>0.118284</td>
          <td>25.799545</td>
          <td>0.235116</td>
          <td>25.185534</td>
          <td>0.303116</td>
          <td>0.011165</td>
          <td>0.007867</td>
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
          <td>27.916033</td>
          <td>1.090628</td>
          <td>26.836662</td>
          <td>0.212031</td>
          <td>25.377237</td>
          <td>0.053582</td>
          <td>25.034599</td>
          <td>0.065055</td>
          <td>24.783759</td>
          <td>0.098813</td>
          <td>24.743840</td>
          <td>0.211375</td>
          <td>0.030979</td>
          <td>0.016659</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.249463</td>
          <td>0.718424</td>
          <td>26.696435</td>
          <td>0.188735</td>
          <td>26.006886</td>
          <td>0.093621</td>
          <td>25.124391</td>
          <td>0.070547</td>
          <td>24.822262</td>
          <td>0.102354</td>
          <td>24.263565</td>
          <td>0.140783</td>
          <td>0.037964</td>
          <td>0.026094</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.980719</td>
          <td>0.278701</td>
          <td>26.776981</td>
          <td>0.202370</td>
          <td>26.608884</td>
          <td>0.158207</td>
          <td>26.241357</td>
          <td>0.186587</td>
          <td>25.886912</td>
          <td>0.253992</td>
          <td>25.069283</td>
          <td>0.277426</td>
          <td>0.052501</td>
          <td>0.027499</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.882495</td>
          <td>0.259439</td>
          <td>26.195974</td>
          <td>0.124523</td>
          <td>26.192037</td>
          <td>0.111638</td>
          <td>25.794163</td>
          <td>0.128759</td>
          <td>25.799684</td>
          <td>0.239001</td>
          <td>25.400553</td>
          <td>0.365239</td>
          <td>0.076657</td>
          <td>0.068339</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.257376</td>
          <td>0.721840</td>
          <td>26.794138</td>
          <td>0.204712</td>
          <td>26.479245</td>
          <td>0.141087</td>
          <td>26.855610</td>
          <td>0.308587</td>
          <td>25.790935</td>
          <td>0.233950</td>
          <td>25.598388</td>
          <td>0.419819</td>
          <td>0.033107</td>
          <td>0.020914</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



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
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.562482</td>
          <td>0.155856</td>
          <td>26.042169</td>
          <td>0.088258</td>
          <td>25.113842</td>
          <td>0.063602</td>
          <td>24.659260</td>
          <td>0.081070</td>
          <td>24.048556</td>
          <td>0.106600</td>
          <td>0.086743</td>
          <td>0.067246</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.604372</td>
          <td>0.346143</td>
          <td>26.875519</td>
          <td>0.168811</td>
          <td>26.612109</td>
          <td>0.215603</td>
          <td>25.721200</td>
          <td>0.188828</td>
          <td>25.185396</td>
          <td>0.259760</td>
          <td>0.000121</td>
          <td>0.000083</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.799425</td>
          <td>0.820357</td>
          <td>27.451356</td>
          <td>0.273025</td>
          <td>25.831848</td>
          <td>0.110685</td>
          <td>24.828987</td>
          <td>0.087364</td>
          <td>24.233486</td>
          <td>0.116004</td>
          <td>0.008646</td>
          <td>0.005456</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.985728</td>
          <td>0.584107</td>
          <td>28.058083</td>
          <td>0.536484</td>
          <td>28.193484</td>
          <td>0.539677</td>
          <td>26.199082</td>
          <td>0.172300</td>
          <td>25.274839</td>
          <td>0.145517</td>
          <td>24.710113</td>
          <td>0.197573</td>
          <td>0.120120</td>
          <td>0.081663</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.365708</td>
          <td>0.339654</td>
          <td>26.358742</td>
          <td>0.122701</td>
          <td>25.967883</td>
          <td>0.076739</td>
          <td>25.655891</td>
          <td>0.094941</td>
          <td>25.293384</td>
          <td>0.131132</td>
          <td>24.669889</td>
          <td>0.169045</td>
          <td>0.011165</td>
          <td>0.007867</td>
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
          <td>27.026960</td>
          <td>0.562118</td>
          <td>26.354945</td>
          <td>0.123040</td>
          <td>25.379115</td>
          <td>0.045855</td>
          <td>25.127312</td>
          <td>0.059965</td>
          <td>24.933312</td>
          <td>0.096482</td>
          <td>24.584754</td>
          <td>0.158322</td>
          <td>0.030979</td>
          <td>0.016659</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.066860</td>
          <td>0.580327</td>
          <td>26.651130</td>
          <td>0.159577</td>
          <td>26.131662</td>
          <td>0.089817</td>
          <td>25.076943</td>
          <td>0.057703</td>
          <td>24.809163</td>
          <td>0.087019</td>
          <td>24.156884</td>
          <td>0.110030</td>
          <td>0.037964</td>
          <td>0.026094</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>30.325150</td>
          <td>2.960839</td>
          <td>26.600284</td>
          <td>0.153953</td>
          <td>26.406841</td>
          <td>0.115305</td>
          <td>26.253213</td>
          <td>0.162996</td>
          <td>26.093724</td>
          <td>0.263042</td>
          <td>25.764551</td>
          <td>0.420009</td>
          <td>0.052501</td>
          <td>0.027499</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.823828</td>
          <td>0.228587</td>
          <td>26.161869</td>
          <td>0.109433</td>
          <td>25.982532</td>
          <td>0.083029</td>
          <td>25.985181</td>
          <td>0.135389</td>
          <td>25.845928</td>
          <td>0.223386</td>
          <td>25.044385</td>
          <td>0.246817</td>
          <td>0.076657</td>
          <td>0.068339</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.939608</td>
          <td>0.528295</td>
          <td>26.782521</td>
          <td>0.177868</td>
          <td>26.459881</td>
          <td>0.119225</td>
          <td>26.187641</td>
          <td>0.152098</td>
          <td>26.078812</td>
          <td>0.256770</td>
          <td>26.659979</td>
          <td>0.785865</td>
          <td>0.033107</td>
          <td>0.020914</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
