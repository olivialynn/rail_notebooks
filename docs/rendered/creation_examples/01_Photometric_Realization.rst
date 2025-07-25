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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fa7645f5ba0>



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
    0      23.994413  0.083384  0.080511  
    1      25.391064  0.096452  0.072811  
    2      24.304707  0.081409  0.072048  
    3      25.291103  0.087530  0.071937  
    4      25.096743  0.076252  0.065103  
    ...          ...       ...       ...  
    99995  24.737946  0.066407  0.034428  
    99996  24.224169  0.151115  0.150763  
    99997  25.613836  0.006235  0.005266  
    99998  25.274899  0.225255  0.164431  
    99999  25.699642  0.057150  0.049113  
    
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
          <td>27.005161</td>
          <td>0.550755</td>
          <td>26.835718</td>
          <td>0.184464</td>
          <td>25.969853</td>
          <td>0.076775</td>
          <td>25.128597</td>
          <td>0.059506</td>
          <td>24.600267</td>
          <td>0.071342</td>
          <td>24.042020</td>
          <td>0.098061</td>
          <td>0.083384</td>
          <td>0.080511</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.812744</td>
          <td>0.478308</td>
          <td>27.623135</td>
          <td>0.351292</td>
          <td>26.448744</td>
          <td>0.116878</td>
          <td>26.020849</td>
          <td>0.130349</td>
          <td>25.629148</td>
          <td>0.174671</td>
          <td>25.316858</td>
          <td>0.289068</td>
          <td>0.096452</td>
          <td>0.072811</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.500543</td>
          <td>0.775336</td>
          <td>27.692339</td>
          <td>0.370849</td>
          <td>28.218829</td>
          <td>0.495980</td>
          <td>25.958470</td>
          <td>0.123489</td>
          <td>25.049676</td>
          <td>0.105951</td>
          <td>24.213485</td>
          <td>0.113917</td>
          <td>0.081409</td>
          <td>0.072048</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.287047</td>
          <td>2.015468</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.820171</td>
          <td>0.366221</td>
          <td>26.395918</td>
          <td>0.179760</td>
          <td>25.552512</td>
          <td>0.163639</td>
          <td>24.673038</td>
          <td>0.169282</td>
          <td>0.087530</td>
          <td>0.071937</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.601263</td>
          <td>0.407676</td>
          <td>26.152895</td>
          <td>0.102450</td>
          <td>26.066958</td>
          <td>0.083644</td>
          <td>25.728620</td>
          <td>0.101059</td>
          <td>25.462124</td>
          <td>0.151462</td>
          <td>24.799287</td>
          <td>0.188405</td>
          <td>0.076252</td>
          <td>0.065103</td>
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
          <td>29.179349</td>
          <td>1.925329</td>
          <td>26.322164</td>
          <td>0.118738</td>
          <td>25.554173</td>
          <td>0.053117</td>
          <td>25.052948</td>
          <td>0.055642</td>
          <td>24.941417</td>
          <td>0.096368</td>
          <td>24.766082</td>
          <td>0.183191</td>
          <td>0.066407</td>
          <td>0.034428</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.604332</td>
          <td>0.829512</td>
          <td>26.900292</td>
          <td>0.194787</td>
          <td>25.883151</td>
          <td>0.071109</td>
          <td>25.131864</td>
          <td>0.059679</td>
          <td>24.817961</td>
          <td>0.086458</td>
          <td>24.352085</td>
          <td>0.128493</td>
          <td>0.151115</td>
          <td>0.150763</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.569520</td>
          <td>0.811065</td>
          <td>26.667918</td>
          <td>0.159953</td>
          <td>26.668849</td>
          <td>0.141423</td>
          <td>26.337975</td>
          <td>0.171131</td>
          <td>26.131465</td>
          <td>0.265517</td>
          <td>25.120300</td>
          <td>0.246246</td>
          <td>0.006235</td>
          <td>0.005266</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.648445</td>
          <td>0.422645</td>
          <td>26.201198</td>
          <td>0.106866</td>
          <td>26.107745</td>
          <td>0.086704</td>
          <td>25.802658</td>
          <td>0.107820</td>
          <td>25.859491</td>
          <td>0.212083</td>
          <td>25.992370</td>
          <td>0.488420</td>
          <td>0.225255</td>
          <td>0.164431</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.064405</td>
          <td>0.574690</td>
          <td>26.707156</td>
          <td>0.165398</td>
          <td>26.416824</td>
          <td>0.113674</td>
          <td>26.362295</td>
          <td>0.174705</td>
          <td>25.917527</td>
          <td>0.222596</td>
          <td>25.436603</td>
          <td>0.318238</td>
          <td>0.057150</td>
          <td>0.049113</td>
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
          <td>29.463819</td>
          <td>2.299613</td>
          <td>26.430161</td>
          <td>0.152999</td>
          <td>25.943072</td>
          <td>0.090196</td>
          <td>25.295754</td>
          <td>0.083691</td>
          <td>24.686339</td>
          <td>0.092575</td>
          <td>23.875397</td>
          <td>0.102427</td>
          <td>0.083384</td>
          <td>0.080511</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.235257</td>
          <td>0.299545</td>
          <td>26.475231</td>
          <td>0.143612</td>
          <td>26.122723</td>
          <td>0.171847</td>
          <td>25.586137</td>
          <td>0.201321</td>
          <td>25.452280</td>
          <td>0.382532</td>
          <td>0.096452</td>
          <td>0.072811</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.359305</td>
          <td>1.402651</td>
          <td>28.976568</td>
          <td>1.025374</td>
          <td>28.175581</td>
          <td>0.558546</td>
          <td>26.281975</td>
          <td>0.195748</td>
          <td>25.054442</td>
          <td>0.127307</td>
          <td>24.506729</td>
          <td>0.176142</td>
          <td>0.081409</td>
          <td>0.072048</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.910815</td>
          <td>0.503455</td>
          <td>27.655770</td>
          <td>0.378880</td>
          <td>26.236743</td>
          <td>0.188736</td>
          <td>25.784486</td>
          <td>0.236849</td>
          <td>24.851100</td>
          <td>0.235457</td>
          <td>0.087530</td>
          <td>0.071937</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.538499</td>
          <td>0.435082</td>
          <td>26.072643</td>
          <td>0.111792</td>
          <td>25.894318</td>
          <td>0.085931</td>
          <td>25.778946</td>
          <td>0.126968</td>
          <td>25.688418</td>
          <td>0.217762</td>
          <td>24.635697</td>
          <td>0.195859</td>
          <td>0.076252</td>
          <td>0.065103</td>
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
          <td>27.022952</td>
          <td>0.616932</td>
          <td>26.770301</td>
          <td>0.201843</td>
          <td>25.329939</td>
          <td>0.051762</td>
          <td>25.059259</td>
          <td>0.067001</td>
          <td>24.681434</td>
          <td>0.090990</td>
          <td>24.374452</td>
          <td>0.155754</td>
          <td>0.066407</td>
          <td>0.034428</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.290216</td>
          <td>0.373254</td>
          <td>26.520437</td>
          <td>0.172645</td>
          <td>26.078949</td>
          <td>0.106760</td>
          <td>25.136057</td>
          <td>0.076535</td>
          <td>24.783867</td>
          <td>0.105945</td>
          <td>24.349627</td>
          <td>0.162307</td>
          <td>0.151115</td>
          <td>0.150763</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.593869</td>
          <td>0.448724</td>
          <td>26.768918</td>
          <td>0.199999</td>
          <td>26.422402</td>
          <td>0.134007</td>
          <td>26.105719</td>
          <td>0.165332</td>
          <td>25.892792</td>
          <td>0.253838</td>
          <td>25.059446</td>
          <td>0.273699</td>
          <td>0.006235</td>
          <td>0.005266</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.827747</td>
          <td>0.266565</td>
          <td>26.418236</td>
          <td>0.164595</td>
          <td>26.110785</td>
          <td>0.114669</td>
          <td>25.791951</td>
          <td>0.141973</td>
          <td>25.673744</td>
          <td>0.236224</td>
          <td>25.431328</td>
          <td>0.408728</td>
          <td>0.225255</td>
          <td>0.164431</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.300262</td>
          <td>0.360393</td>
          <td>26.391518</td>
          <td>0.146350</td>
          <td>26.661558</td>
          <td>0.166070</td>
          <td>26.052332</td>
          <td>0.159478</td>
          <td>25.943269</td>
          <td>0.266877</td>
          <td>25.763217</td>
          <td>0.478314</td>
          <td>0.057150</td>
          <td>0.049113</td>
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
          <td>27.170731</td>
          <td>0.261255</td>
          <td>25.951807</td>
          <td>0.082188</td>
          <td>25.148544</td>
          <td>0.066165</td>
          <td>24.511923</td>
          <td>0.071775</td>
          <td>23.951377</td>
          <td>0.098752</td>
          <td>0.083384</td>
          <td>0.080511</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.746247</td>
          <td>3.410698</td>
          <td>27.072694</td>
          <td>0.242290</td>
          <td>26.474225</td>
          <td>0.130562</td>
          <td>26.427951</td>
          <td>0.202068</td>
          <td>25.936614</td>
          <td>0.246037</td>
          <td>25.777878</td>
          <td>0.450243</td>
          <td>0.096452</td>
          <td>0.072811</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.090518</td>
          <td>0.610777</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.518857</td>
          <td>0.308447</td>
          <td>26.115464</td>
          <td>0.152606</td>
          <td>25.140486</td>
          <td>0.123418</td>
          <td>24.244822</td>
          <td>0.126318</td>
          <td>0.081409</td>
          <td>0.072048</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.565820</td>
          <td>0.416867</td>
          <td>28.364720</td>
          <td>0.646312</td>
          <td>27.831950</td>
          <td>0.396606</td>
          <td>26.244074</td>
          <td>0.171314</td>
          <td>25.411473</td>
          <td>0.156774</td>
          <td>25.122604</td>
          <td>0.266518</td>
          <td>0.087530</td>
          <td>0.071937</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.960093</td>
          <td>0.553210</td>
          <td>26.154746</td>
          <td>0.108474</td>
          <td>25.968729</td>
          <td>0.081781</td>
          <td>25.847864</td>
          <td>0.119835</td>
          <td>25.402187</td>
          <td>0.153105</td>
          <td>24.805335</td>
          <td>0.201749</td>
          <td>0.076252</td>
          <td>0.065103</td>
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
          <td>29.942579</td>
          <td>2.617442</td>
          <td>26.385648</td>
          <td>0.129426</td>
          <td>25.457030</td>
          <td>0.050546</td>
          <td>24.972303</td>
          <td>0.053824</td>
          <td>24.708740</td>
          <td>0.081426</td>
          <td>24.860336</td>
          <td>0.205583</td>
          <td>0.066407</td>
          <td>0.034428</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.636059</td>
          <td>0.487188</td>
          <td>26.694285</td>
          <td>0.200857</td>
          <td>25.960719</td>
          <td>0.096772</td>
          <td>25.241854</td>
          <td>0.084478</td>
          <td>24.937480</td>
          <td>0.121744</td>
          <td>24.068810</td>
          <td>0.128162</td>
          <td>0.151115</td>
          <td>0.150763</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.792223</td>
          <td>0.177865</td>
          <td>26.590828</td>
          <td>0.132271</td>
          <td>26.029192</td>
          <td>0.131355</td>
          <td>26.113598</td>
          <td>0.261779</td>
          <td>25.396859</td>
          <td>0.308418</td>
          <td>0.006235</td>
          <td>0.005266</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.836245</td>
          <td>0.283571</td>
          <td>26.282448</td>
          <td>0.156493</td>
          <td>26.013899</td>
          <td>0.113225</td>
          <td>25.622498</td>
          <td>0.131914</td>
          <td>25.743772</td>
          <td>0.267666</td>
          <td>25.592271</td>
          <td>0.491524</td>
          <td>0.225255</td>
          <td>0.164431</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.902080</td>
          <td>0.522122</td>
          <td>26.741210</td>
          <td>0.175666</td>
          <td>26.600368</td>
          <td>0.138257</td>
          <td>26.340572</td>
          <td>0.178077</td>
          <td>25.988237</td>
          <td>0.244395</td>
          <td>26.725663</td>
          <td>0.837059</td>
          <td>0.057150</td>
          <td>0.049113</td>
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
