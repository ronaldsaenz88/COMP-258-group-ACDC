$(document).ready(function () {
    $("#predictionForm").submit(function (event) {
        event.preventDefault();
        let formData = $(this).serializeArray();
        
        const firstTermGPA = $('#firstTermGPA').val();
        const secondTermGPA = $('#secondTermGPA').val();
        const firstLanguage = $('#firstLanguage').val();
        const funding = $('#funding').val();
        const school = $('#school').val();
        const fastTrack = $('#fastTrack').val();
        const coop = $('#coop').val();
        const residency = $('#residency').val();
        const gender = $('#gender').val();
        const previousEducation = $('#previousEducation').val();
        const ageGroup = $('#ageGroup').val();
        const highSchoolAverageMark = $('#highSchoolAverageMark').val();
        const mathScore = $('#mathScore').val();
        const englishGrade = $('#englishGrade').val();

        var dataJson = JSON.stringify( [{
            "FirstTermGpa": parseFloat(firstTermGPA),
            "SecondTermGpa": parseFloat(secondTermGPA),
            "FirstLanguage": parseFloat(firstLanguage),
            "Funding": parseFloat(funding),
            "School": parseFloat(school),
            "FastTrack": parseFloat(fastTrack),
            "Coop": parseFloat(coop),
            "Residency": parseFloat(residency),
            "Gender": parseFloat(gender),
            "PreviousEducation": parseFloat(previousEducation),
            "AgeGroup": parseFloat(ageGroup),
            "HighSchoolAverageMark": parseFloat(highSchoolAverageMark),
            "MathScore": parseFloat(mathScore),
            "EnglishGrade": parseFloat(englishGrade)
        }] );

        // Show the modal and loading spinner
        $("#resultModal").modal("show");
        $("#resultLoading").show();
        $("#resultGood").hide().removeClass("animate-scale-up");
        $("#resultSad").hide().removeClass("animate-scale-up");

        // Simulate a short delay (e.g., waiting for API response)
        setTimeout(function () {
            $.ajax({
                url: "http://127.0.0.1:5000/api/predict",
                dataType: 'json',
                type: 'POST',
                contentType: 'application/json',
                data: dataJson,
                processData: false,
                success: function (result,status,xhr) {
                    mockPrediction = result.output[0]
                            
                    // Update the result text
                    $("#resultText").html("<b>The predicted First Year Persistence is:</b> <br>" + mockPrediction);
            
                    // Hide the loading spinner and show the appropriate icon and animate it
                    $("#resultLoading").hide();
                    if (mockPrediction === 1) {
                        $("#resultGood").show().addClass("animate-scale-up");
                        $("#resultSad").hide();
                    } else {
                        $("#resultSad").show().addClass("animate-scale-up");
                        $("#resultGood").hide();
                    }
                }
            });        
        }, 1000); // 1000 ms (1 second) delay to simulate waiting for the API response
    });


    
    $("#btn_summary").click(function (event) {
        event.preventDefault();

        // Show the modal and loading spinner
        $("#resultModal").modal("show");
        $("#resultLoading").show();
        $("#resultGood").hide().removeClass("animate-scale-up");
        $("#resultSad").hide().removeClass("animate-scale-up");

        // Simulate a short delay (e.g., waiting for API response)
        setTimeout(function () {
            $.ajax({
                url: "http://127.0.0.1:5000/api/summary",
                dataType: 'json',
                type: 'GET',
                contentType: 'application/json',
                processData: false,
                success: function (result,status,xhr) {
                    mockPrediction = result.output
                            
                    // Update the result text
                    $("#resultText").html("<b>The summary of the model:</b> <br>" + mockPrediction.replace(/\n/g, "<br>"));
            
                    // Hide the loading spinner and show the appropriate icon and animate it
                    $("#resultLoading").hide();
                    if (mockPrediction !== null) {
                        $("#resultGood").show().addClass("animate-scale-up");
                        $("#resultSad").hide();
                    } else {
                        $("#resultSad").show().addClass("animate-scale-up");
                        $("#resultGood").hide();
                    }
                }
            });        
        }, 1000); // 1000 ms (1 second) delay to simulate waiting for the API response
    });

    $("#btn_scores").click(function (event) {
        event.preventDefault();

        // Show the modal and loading spinner
        $("#resultModal").modal("show");
        $("#resultLoading").show();
        $("#resultGood").hide().removeClass("animate-scale-up");
        $("#resultSad").hide().removeClass("animate-scale-up");

        // Simulate a short delay (e.g., waiting for API response)
        setTimeout(function () {
            $.ajax({
                url: "http://127.0.0.1:5000/api/scores",
                dataType: 'json',
                type: 'GET',
                contentType: 'application/json',
                processData: false,
                success: function (result,status,xhr) {
                    mockPrediction = result.output

                    // Update the result text
                    $("#resultText").html(
                        "<b>Test accuracy:</b> " + mockPrediction.accuracy + "<br>" +
                        "<b>Test precision:</b> " + mockPrediction.precision + "<br>" +
                        "<b>Test recall:</b> " + mockPrediction.recall + "<br>" +
                        "<b>Test f1:</b> " + mockPrediction.f1 + "<br>" +
                        "<b>Test roc_auc:</b> " + mockPrediction.roc_auc + "<br>" +
                        "<b>Test confussion_matrix:</b> <br>" + mockPrediction.confussion_matrix.replace(/\n/g, "<br>") + "<br>" +
                        "<b>Test classification_report_var:</b> <br>" + mockPrediction.classification_report_var.replace(/\n/g, "<br>") + "<br>"
                    );

                    // Hide the loading spinner and show the appropriate icon and animate it
                    $("#resultLoading").hide();
                    if (mockPrediction !== null) {
                        $("#resultGood").show().addClass("animate-scale-up");
                        $("#resultSad").hide();
                    } else {
                        $("#resultSad").show().addClass("animate-scale-up");
                        $("#resultGood").hide();
                    }
                }
            });        
        }, 1000); // 1000 ms (1 second) delay to simulate waiting for the API response
    });

});
  