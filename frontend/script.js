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
            "FirstTermGpa": firstTermGPA,
            "SecondTermGpa": secondTermGPA,
            "FirstLanguage": firstLanguage,
            "Funding": funding,
            "School": school,
            "FastTrack": fastTrack,
            "Coop": coop,
            "Residency": residency,
            "Gender": gender,
            "PreviousEducation": previousEducation,
            "AgeGroup": ageGroup,
            "HighSchoolAverageMark": highSchoolAverageMark,
            "MathScore": mathScore,
            "EnglishGrade": englishGrade
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
                    $("#resultText").text("The predicted First Year Persistence is: " + mockPrediction);
            
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

            /*
            // Set a fixed mock prediction result
            const mockPrediction = 1;
            
            alert(formData);

            // Update the result text
            $("#resultText").text("The predicted First Year Persistence is: " + mockPrediction);
    
            // Hide the loading spinner and show the appropriate icon and animate it
            $("#resultLoading").hide();
            if (mockPrediction === 1) {
            $("#resultGood").show().addClass("animate-scale-up");
            $("#resultSad").hide();
            } else {
            $("#resultSad").show().addClass("animate-scale-up");
            $("#resultGood").hide();
            }*/
        
        }, 1000); // 1000 ms (1 second) delay to simulate waiting for the API response
    });
});
  